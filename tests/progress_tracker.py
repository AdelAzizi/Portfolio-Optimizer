#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سیستم ردیابی پیشرفت و مدیریت خطاها

این ماژول شامل کلاس‌های مدیریت پیشرفت، لاگینگ پیشرفته و مدیریت خطاها است.
"""

import logging
import sys
import time
import traceback
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log levels for the system"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TaskStatus(Enum):
    """Task status enumeration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskInfo:
    """Information about a task"""
    name: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    sub_tasks: List['TaskInfo'] = None
    
    def __post_init__(self):
        if self.sub_tasks is None:
            self.sub_tasks = []


class ProgressTracker:
    """
    کلاس ردیابی پیشرفت real-time برای تحلیل‌ها
    """
    
    def __init__(self, total_tasks: int = 0, 
                 log_file: str = "tests/results/analysis.log",
                 console_output: bool = True):
        """
        مقداردهی اولیه
        
        Args:
            total_tasks: تعداد کل تسک‌ها
            log_file: مسیر فایل لاگ
            console_output: نمایش خروجی در کنسول
        """
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.current_task = None
        self.start_time = None
        self.tasks_info: Dict[str, TaskInfo] = {}
        self.lock = threading.Lock()
        
        # تنظیم لاگینگ
        self.logger = self._setup_logging(log_file, console_output)
        
        self.logger.info("Progress tracking system initialized")
    
    def _setup_logging(self, log_file: str, console_output: bool) -> logging.Logger:
        """تنظیم سیستم لاگینگ"""
        logger = logging.getLogger('ProgressTracker')
        logger.setLevel(logging.INFO)
        
        # حذف handler های قبلی
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # فرمت لاگ
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def start_tracking(self):
        """شروع ردیابی پیشرفت"""
        with self.lock:
            self.start_time = datetime.now()
            self.logger.info(f"🚀 Starting analysis with {self.total_tasks} tasks")
    
    def start_task(self, task_name: str, estimated_duration: Optional[float] = None):
        """
        شروع یک تسک
        
        Args:
            task_name: نام تسک
            estimated_duration: مدت زمان تخمینی (ثانیه)
        """
        with self.lock:
            self.current_task = task_name
            
            task_info = TaskInfo(
                name=task_name,
                status=TaskStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
            
            self.tasks_info[task_name] = task_info
            
            progress = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
            
            self.logger.info(f"▶️ Starting task: {task_name} (Progress: {progress:.1f}%)")
    
    def update_task_progress(self, task_name: str, progress: float, 
                           message: Optional[str] = None):
        """
        به‌روزرسانی پیشرفت تسک
        
        Args:
            task_name: نام تسک
            progress: درصد پیشرفت (0-100)
            message: پیام اضافی
        """
        with self.lock:
            if task_name in self.tasks_info:
                self.tasks_info[task_name].progress_percentage = progress
                
                if message:
                    self.logger.info(f"📊 {task_name}: {progress:.1f}% - {message}")
                else:
                    self.logger.debug(f"📊 {task_name}: {progress:.1f}%")
    
    def complete_task(self, task_name: str, success: bool = True, 
                     error_message: Optional[str] = None):
        """
        تکمیل یک تسک
        
        Args:
            task_name: نام تسک
            success: موفقیت تسک
            error_message: پیام خطا در صورت شکست
        """
        with self.lock:
            if task_name in self.tasks_info:
                task_info = self.tasks_info[task_name]
                task_info.end_time = datetime.now()
                
                if task_info.start_time:
                    duration = (task_info.end_time - task_info.start_time).total_seconds()
                    task_info.duration = duration
                
                if success:
                    task_info.status = TaskStatus.COMPLETED
                    task_info.progress_percentage = 100.0
                    self.completed_tasks += 1
                    
                    duration_str = f" ({task_info.duration:.1f}s)" if task_info.duration else ""
                    self.logger.info(f"✅ Completed task: {task_name}{duration_str}")
                else:
                    task_info.status = TaskStatus.FAILED
                    task_info.error_message = error_message
                    self.failed_tasks += 1
                    
                    self.logger.error(f"❌ Failed task: {task_name} - {error_message}")
                
                # نمایش پیشرفت کلی
                self._log_overall_progress()
    
    def _log_overall_progress(self):
        """نمایش پیشرفت کلی"""
        if self.total_tasks > 0:
            overall_progress = (self.completed_tasks / self.total_tasks * 100)
            remaining_tasks = self.total_tasks - self.completed_tasks - self.failed_tasks
            
            elapsed_time = ""
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                elapsed_time = f" | Elapsed: {elapsed:.0f}s"
            
            self.logger.info(
                f"📈 Overall Progress: {overall_progress:.1f}% "
                f"({self.completed_tasks}/{self.total_tasks} completed, "
                f"{self.failed_tasks} failed, {remaining_tasks} remaining){elapsed_time}"
            )
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        دریافت خلاصه پیشرفت
        
        Returns:
            دیکشنری حاوی اطلاعات پیشرفت
        """
        with self.lock:
            elapsed_time = None
            if self.start_time:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'total_tasks': self.total_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'remaining_tasks': self.total_tasks - self.completed_tasks - self.failed_tasks,
                'overall_progress': (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0,
                'elapsed_time': elapsed_time,
                'current_task': self.current_task,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'tasks_info': {name: asdict(info) for name, info in self.tasks_info.items()}
            }
    
    def save_progress_report(self, file_path: str = "tests/results/progress_report.json"):
        """
        ذخیره گزارش پیشرفت
        
        Args:
            file_path: مسیر فایل گزارش
        """
        try:
            report = self.get_progress_summary()
            
            # تبدیل datetime objects به string
            for task_name, task_info in report['tasks_info'].items():
                if task_info.get('start_time'):
                    task_info['start_time'] = task_info['start_time'].isoformat()
                if task_info.get('end_time'):
                    task_info['end_time'] = task_info['end_time'].isoformat()
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Progress report saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving progress report: {e}")


class ErrorHandler:
    """
    کلاس مدیریت خطاها و exception handling
    """
    
    def __init__(self, log_file: str = "tests/results/errors.log",
                 max_retries: int = 3):
        """
        مقداردهی اولیه
        
        Args:
            log_file: مسیر فایل لاگ خطاها
            max_retries: حداکثر تعداد تلاش مجدد
        """
        self.max_retries = max_retries
        self.error_count = 0
        self.critical_errors = []
        
        # تنظیم لاگینگ خطاها
        self.logger = self._setup_error_logging(log_file)
        
        self.logger.info("Error handling system initialized")
    
    def _setup_error_logging(self, log_file: str) -> logging.Logger:
        """تنظیم سیستم لاگینگ خطاها"""
        logger = logging.getLogger('ErrorHandler')
        logger.setLevel(logging.ERROR)
        
        # حذف handler های قبلی
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # فرمت لاگ خطاها
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Traceback: %(exc_info)s\n' + '-'*50,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler برای خطاها
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def handle_error(self, error: Exception, context: str = "", 
                    critical: bool = False) -> bool:
        """
        مدیریت خطا
        
        Args:
            error: خطای رخ داده
            context: زمینه رخداد خطا
            critical: آیا خطا حیاتی است
            
        Returns:
            True اگر خطا قابل ادامه باشد
        """
        self.error_count += 1
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        if critical:
            self.critical_errors.append(error_info)
            self.logger.critical(f"CRITICAL ERROR in {context}: {error}", exc_info=True)
            return False
        else:
            self.logger.error(f"ERROR in {context}: {error}", exc_info=True)
            return True
    
    def retry_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        تلاش مجدد برای اجرای عملیات
        
        Args:
            operation: تابع برای اجرا
            *args: آرگومان‌های تابع
            **kwargs: کلیدواژه‌های تابع
            
        Returns:
            نتیجه عملیات یا None در صورت شکست
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    self.handle_error(e, f"Final attempt ({attempt + 1}) failed", critical=True)
        
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        دریافت خلاصه خطاها
        
        Returns:
            دیکشنری حاوی اطلاعات خطاها
        """
        return {
            'total_errors': self.error_count,
            'critical_errors_count': len(self.critical_errors),
            'critical_errors': self.critical_errors,
            'max_retries': self.max_retries
        }


def test_progress_system():
    """
    تست سیستم ردیابی پیشرفت
    """
    print("🧪 Testing progress tracking system...")
    
    # تست ProgressTracker
    tracker = ProgressTracker(total_tasks=3, console_output=True)
    tracker.start_tracking()
    
    # تست تسک اول
    tracker.start_task("Test Task 1")
    time.sleep(1)
    tracker.update_task_progress("Test Task 1", 50, "Processing data...")
    time.sleep(1)
    tracker.complete_task("Test Task 1", success=True)
    
    # تست تسک دوم
    tracker.start_task("Test Task 2")
    time.sleep(0.5)
    tracker.complete_task("Test Task 2", success=False, error_message="Test error")
    
    # تست تسک سوم
    tracker.start_task("Test Task 3")
    time.sleep(0.5)
    tracker.complete_task("Test Task 3", success=True)
    
    # ذخیره گزارش
    tracker.save_progress_report()
    
    # تست ErrorHandler
    error_handler = ErrorHandler()
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_handler.handle_error(e, "Test context")
    
    # تست retry operation
    def failing_operation():
        raise ConnectionError("Connection failed")
    
    result = error_handler.retry_operation(failing_operation)
    print(f"Retry result: {result}")
    
    print("✅ Progress system test completed!")


if __name__ == "__main__":
    test_progress_system()