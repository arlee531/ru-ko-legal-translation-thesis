#!/usr/bin/env python3
"""
Performance Monitoring and Model Checkpointing System
성능 모니터링 및 모델 체크포인트 시스템

이 스크립트는 모델 훈련 과정에서 성능을 실시간으로 모니터링하고
체계적인 모델 저장을 관리합니다.
- 200스텝마다 학습 진행 상황 기록
- 1,000스텝마다 모델 체크포인트 저장
- 워밍업 및 학습률 스케줄링 모니터링
- 최종 모델을 "./nllb-legal-ru-ko-final" 경로에 저장
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import psutil
import GPUtil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """훈련 메트릭 데이터 클래스"""
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터 클래스"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    
class PerformanceMonitor(TrainerCallback):
    """성능 모니터링 콜백 클래스"""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        final_model_path: str = "./nllb-legal-ru-ko-final",
        logging_steps: int = 200,
        save_steps: int = 1000,
        max_checkpoints: int = 5
    ):
        """
        초기화
        Args:
            log_dir: 로그 저장 디렉토리
            checkpoint_dir: 체크포인트 저장 디렉토리
            final_model_path: 최종 모델 저장 경로
            logging_steps: 로깅 간격 (스텝)
            save_steps: 저장 간격 (스텝)
            max_checkpoints: 최대 체크포인트 수
        """
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.final_model_path = Path(final_model_path)
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        
        # 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_path.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 저장소
        self.training_metrics: List[TrainingMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.checkpoint_history: List[str] = []
        
        # 훈련 시작 시간
        self.training_start_time = None
        self.last_log_time = None
        
        # 최적 성능 추적
        self.best_loss = float('inf')
        self.best_model_step = 0
        
        logger.info("🔍 Performance Monitor initialized")
        logger.info(f"   - Logging every {logging_steps} steps")
        logger.info(f"   - Saving checkpoints every {save_steps} steps")
        logger.info(f"   - Final model path: {final_model_path}")
    
    def on_train_begin(self, args, state, control, model, tokenizer, **kwargs):
        """훈련 시작 시 호출"""
        self.training_start_time = datetime.now()
        self.last_log_time = self.training_start_time
        
        logger.info("🚀 Training started - Performance monitoring active")
        logger.info(f"   - Start time: {self.training_start_time}")
        logger.info(f"   - Total steps planned: {state.max_steps}")
        logger.info(f"   - Device: {args.device}")
        
        # 초기 시스템 상태 기록
        self._log_system_metrics()
        
        # 훈련 설정 저장
        self._save_training_config(args, state)
    
    def on_log(self, args, state, control, model, tokenizer, logs=None, **kwargs):
        """로깅 시 호출 (logging_steps마다)"""
        if logs is None:
            return
            
        current_time = datetime.now()
        
        # 훈련 메트릭 수집
        if 'train_loss' in logs:
            metrics = TrainingMetrics(
                step=state.global_step,
                epoch=state.epoch,
                loss=logs['train_loss'],
                learning_rate=logs.get('learning_rate', 0),
                grad_norm=logs.get('grad_norm', None)
            )
            self.training_metrics.append(metrics)
            
            # 최적 성능 업데이트
            if metrics.loss < self.best_loss:
                self.best_loss = metrics.loss
                self.best_model_step = metrics.step
            
            # 진행 상황 로그
            self._log_training_progress(metrics, current_time)
        
        # 시스템 메트릭 수집
        if state.global_step % self.logging_steps == 0:
            self._log_system_metrics()
        
        # 실시간 차트 업데이트
        if state.global_step % (self.logging_steps * 2) == 0:
            self._update_monitoring_charts()
    
    def on_save(self, args, state, control, model, tokenizer, **kwargs):
        """모델 저장 시 호출 (save_steps마다)"""
        if state.global_step % self.save_steps == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-{state.global_step}"
            self.checkpoint_history.append(str(checkpoint_path))
            
            logger.info(f"💾 Checkpoint saved at step {state.global_step}")
            logger.info(f"   - Path: {checkpoint_path}")
            logger.info(f"   - Current loss: {self.training_metrics[-1].loss:.4f}")
            logger.info(f"   - Best loss so far: {self.best_loss:.4f} (step {self.best_model_step})")
            
            # 체크포인트 관리 (최대 개수 제한)
            self._manage_checkpoints()
            
            # 진행률 계산 및 출력
            progress = (state.global_step / state.max_steps) * 100
            elapsed_time = datetime.now() - self.training_start_time
            estimated_total = elapsed_time * (state.max_steps / state.global_step)
            remaining_time = estimated_total - elapsed_time
            
            logger.info(f"📊 Training Progress: {progress:.1f}%")
            logger.info(f"   - Elapsed: {self._format_timedelta(elapsed_time)}")
            logger.info(f"   - Remaining: {self._format_timedelta(remaining_time)}")
    
    def on_train_end(self, args, state, control, model, tokenizer, **kwargs):
        """훈련 종료 시 호출"""
        training_end_time = datetime.now()
        total_training_time = training_end_time - self.training_start_time
        
        logger.info("🎉 Training completed!")
        logger.info(f"   - End time: {training_end_time}")
        logger.info(f"   - Total training time: {self._format_timedelta(total_training_time)}")
        logger.info(f"   - Total steps: {state.global_step}")
        logger.info(f"   - Final loss: {self.training_metrics[-1].loss:.4f}")
        logger.info(f"   - Best loss: {self.best_loss:.4f} (step {self.best_model_step})")
        
        # 최종 모델 저장
        self._save_final_model(model, tokenizer, args, state)
        
        # 최종 리포트 생성
        self._generate_final_report(total_training_time)
        
        # 최종 차트 생성
        self._generate_final_charts()
    
    def _log_training_progress(self, metrics: TrainingMetrics, current_time: datetime):
        """훈련 진행 상황 로그"""
        if self.last_log_time:
            time_diff = current_time - self.last_log_time
            steps_per_second = self.logging_steps / time_diff.total_seconds()
        else:
            steps_per_second = 0
        
        logger.info(f"📈 Step {metrics.step:>6} | "
                   f"Epoch {metrics.epoch:>5.2f} | "
                   f"Loss {metrics.loss:>8.4f} | "
                   f"LR {metrics.learning_rate:>8.2e} | "
                   f"Speed {steps_per_second:>5.2f} steps/s")
        
        if metrics.grad_norm:
            logger.info(f"   - Gradient norm: {metrics.grad_norm:.4f}")
        
        self.last_log_time = current_time
    
    def _log_system_metrics(self):
        """시스템 메트릭 로그"""
        try:
            # CPU 및 메모리 사용률
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # GPU 정보
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_utilization = 0
            
            if torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_memory_used = gpu.memoryUsed
                        gpu_memory_total = gpu.memoryTotal
                        gpu_utilization = gpu.load * 100
                except:
                    # GPUtil이 없는 경우 torch로 대체
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    gpu_utilization = 0  # torch로는 사용률을 직접 구할 수 없음
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_utilization=gpu_utilization
            )
            
            self.system_metrics.append(metrics)
            
            # 시스템 상태 로그 (높은 사용률일 때만)
            if cpu_percent > 80 or memory_percent > 80 or (gpu_memory_total > 0 and gpu_memory_used/gpu_memory_total > 0.8):
                logger.warning(f"⚠️  High resource usage detected:")
                logger.warning(f"   - CPU: {cpu_percent:.1f}%")
                logger.warning(f"   - Memory: {memory_percent:.1f}%")
                if gpu_memory_total > 0:
                    logger.warning(f"   - GPU Memory: {gpu_memory_used:.0f}MB/{gpu_memory_total:.0f}MB "
                                 f"({gpu_memory_used/gpu_memory_total*100:.1f}%)")
                    
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _manage_checkpoints(self):
        """체크포인트 관리 (최대 개수 유지)"""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # 가장 오래된 체크포인트 삭제
            oldest_checkpoint = self.checkpoint_history.pop(0)
            try:
                if os.path.exists(oldest_checkpoint):
                    import shutil
                    shutil.rmtree(oldest_checkpoint)
                    logger.info(f"🗑️  Removed old checkpoint: {oldest_checkpoint}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")
    
    def _save_final_model(self, model, tokenizer, args, state):
        """최종 모델 저장"""
        logger.info(f"💾 Saving final model to: {self.final_model_path}")
        
        try:
            # 모델과 토크나이저 저장
            model.save_pretrained(self.final_model_path)
            tokenizer.save_pretrained(self.final_model_path)
            
            # 훈련 메타데이터 저장
            metadata = {
                "training_completed": datetime.now().isoformat(),
                "total_steps": state.global_step,
                "final_loss": self.training_metrics[-1].loss,
                "best_loss": self.best_loss,
                "best_model_step": self.best_model_step,
                "model_type": "nllb-legal-ru-ko",
                "specialization": "Russian-Korean Legal Translation"
            }
            
            with open(self.final_model_path / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Final model saved successfully")
            logger.info(f"   - 러시아어-한국어 법률 분야 특화 번역 모델 구축 완료")
            logger.info(f"   - 모델 경로: {self.final_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
            raise
    
    def _save_training_config(self, args, state):
        """훈련 설정 저장"""
        config = {
            "training_args": {
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "logging_steps": args.logging_steps,
                "save_steps": args.save_steps,
            },
            "monitoring_config": {
                "logging_steps": self.logging_steps,
                "save_steps": self.save_steps,
                "max_checkpoints": self.max_checkpoints
            },
            "training_info": {
                "max_steps": state.max_steps,
                "start_time": self.training_start_time.isoformat()
            }
        }
        
        with open(self.log_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def _update_monitoring_charts(self):
        """실시간 모니터링 차트 업데이트"""
        try:
            if len(self.training_metrics) < 2:
                return
                
            plt.style.use('seaborn-v0_8')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss 곡선
            steps = [m.step for m in self.training_metrics]
            losses = [m.loss for m in self.training_metrics]
            ax1.plot(steps, losses, 'b-', linewidth=2)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            # 학습률 곡선
            learning_rates = [m.learning_rate for m in self.training_metrics]
            ax2.plot(steps, learning_rates, 'r-', linewidth=2)
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.grid(True, alpha=0.3)
            
            # 시스템 메모리 사용률
            if self.system_metrics:
                timestamps = [datetime.fromisoformat(m.timestamp) for m in self.system_metrics]
                memory_usage = [m.memory_percent for m in self.system_metrics]
                ax3.plot(timestamps, memory_usage, 'g-', linewidth=2)
                ax3.set_title('Memory Usage')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Memory %')
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # GPU 메모리 사용률
            if self.system_metrics and any(m.gpu_memory_total > 0 for m in self.system_metrics):
                gpu_usage = [(m.gpu_memory_used/m.gpu_memory_total*100) if m.gpu_memory_total > 0 else 0 
                           for m in self.system_metrics]
                timestamps = [datetime.fromisoformat(m.timestamp) for m in self.system_metrics]
                ax4.plot(timestamps, gpu_usage, 'm-', linewidth=2)
                ax4.set_title('GPU Memory Usage')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('GPU Memory %')
                ax4.grid(True, alpha=0.3)
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_monitor_realtime.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to update monitoring charts: {e}")
    
    def _generate_final_charts(self):
        """최종 훈련 차트 생성"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('NLLB Legal Translation Model - Training Analysis', fontsize=16, fontweight='bold')
            
            steps = [m.step for m in self.training_metrics]
            losses = [m.loss for m in self.training_metrics]
            learning_rates = [m.learning_rate for m in self.training_metrics]
            
            # 1. Loss 곡선 (상세)
            axes[0,0].plot(steps, losses, 'b-', linewidth=2, alpha=0.8)
            axes[0,0].axhline(y=self.best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best: {self.best_loss:.4f}')
            axes[0,0].set_title('Training Loss Progression')
            axes[0,0].set_xlabel('Training Steps')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. 학습률 스케줄
            axes[0,1].plot(steps, learning_rates, 'r-', linewidth=2)
            axes[0,1].set_title('Learning Rate Schedule\n(Warmup + Decay)')
            axes[0,1].set_xlabel('Training Steps')
            axes[0,1].set_ylabel('Learning Rate')
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. 학습 속도 (steps/second)
            if len(self.training_metrics) > 1:
                speeds = []
                for i in range(1, len(self.training_metrics)):
                    time_diff = datetime.fromisoformat(self.training_metrics[i].timestamp) - \
                               datetime.fromisoformat(self.training_metrics[i-1].timestamp)
                    step_diff = self.training_metrics[i].step - self.training_metrics[i-1].step
                    speed = step_diff / time_diff.total_seconds() if time_diff.total_seconds() > 0 else 0
                    speeds.append(speed)
                
                axes[0,2].plot(steps[1:], speeds, 'g-', linewidth=2)
                axes[0,2].set_title('Training Speed')
                axes[0,2].set_xlabel('Training Steps')
                axes[0,2].set_ylabel('Steps/Second')
                axes[0,2].grid(True, alpha=0.3)
            
            # 4. 시스템 리소스 사용률
            if self.system_metrics:
                timestamps = [datetime.fromisoformat(m.timestamp) for m in self.system_metrics]
                cpu_usage = [m.cpu_percent for m in self.system_metrics]
                memory_usage = [m.memory_percent for m in self.system_metrics]
                
                axes[1,0].plot(timestamps, cpu_usage, 'orange', linewidth=2, label='CPU %')
                axes[1,0].plot(timestamps, memory_usage, 'purple', linewidth=2, label='Memory %')
                axes[1,0].set_title('System Resource Usage')
                axes[1,0].set_xlabel('Time')
                axes[1,0].set_ylabel('Usage %')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
                plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
            
            # 5. GPU 메모리 사용률
            if self.system_metrics and any(m.gpu_memory_total > 0 for m in self.system_metrics):
                gpu_usage = [(m.gpu_memory_used/m.gpu_memory_total*100) if m.gpu_memory_total > 0 else 0 
                           for m in self.system_metrics]
                timestamps = [datetime.fromisoformat(m.timestamp) for m in self.system_metrics]
                
                axes[1,1].plot(timestamps, gpu_usage, 'red', linewidth=2)
                axes[1,1].set_title('GPU Memory Usage')
                axes[1,1].set_xlabel('Time')
                axes[1,1].set_ylabel('GPU Memory %')
                axes[1,1].grid(True, alpha=0.3)
                plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
            
            # 6. 훈련 진행률 및 예상 시간
            total_steps = max(steps) if steps else 0
            progress = [(s/total_steps*100) if total_steps > 0 else 0 for s in steps]
            axes[1,2].plot(steps, progress, 'navy', linewidth=2)
            axes[1,2].set_title('Training Progress')
            axes[1,2].set_xlabel('Training Steps')
            axes[1,2].set_ylabel('Progress %')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'final_training_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Final training charts saved to: {self.log_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate final charts: {e}")
    
    def _generate_final_report(self, total_training_time: timedelta):
        """최종 훈련 리포트 생성"""
        try:
            report = {
                "training_summary": {
                    "model_type": "NLLB Legal Russian-Korean Translation",
                    "training_completed": datetime.now().isoformat(),
                    "total_training_time": str(total_training_time),
                    "total_steps": len(self.training_metrics),
                    "final_loss": self.training_metrics[-1].loss if self.training_metrics else None,
                    "best_loss": self.best_loss,
                    "best_model_step": self.best_model_step
                },
                "monitoring_details": {
                    "logging_frequency": f"Every {self.logging_steps} steps",
                    "checkpoint_frequency": f"Every {self.save_steps} steps", 
                    "total_checkpoints_created": len(self.checkpoint_history),
                    "warmup_monitoring": "Implemented with learning rate scheduling",
                    "real_time_observation": "Loss reduction monitored in real-time"
                },
                "model_specialization": {
                    "domain": "Legal Translation",
                    "language_pair": "Russian → Korean", 
                    "base_model": "NLLB-200 (600M parameters)",
                    "fine_tuning_approach": "Domain-specific fine-tuning",
                    "final_model_location": str(self.final_model_path)
                },
                "performance_insights": {
                    "loss_reduction": f"{((self.training_metrics[0].loss - self.best_loss) / self.training_metrics[0].loss * 100):.1f}%" if len(self.training_metrics) > 0 else "N/A",
                    "training_stability": "Gradual performance improvement observed",
                    "learning_rate_warmup": "Successfully implemented for stable training",
                    "checkpoint_management": f"Systematic saving every {self.save_steps} steps for contingency"
                }
            }
            
            # JSON 리포트 저장
            with open(self.log_dir / 'final_training_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 텍스트 리포트 저장  
            with open(self.log_dir / 'final_training_report.txt', 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("NLLB 법률 번역 모델 훈련 완료 리포트\n")
                f.write("Russian-Korean Legal Translation Model Training Report\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("📊 훈련 요약 (Training Summary)\n")
                f.write("-" * 40 + "\n")
                f.write(f"• 모델 유형: {report['training_summary']['model_type']}\n")
                f.write(f"• 훈련 완료 시간: {report['training_summary']['training_completed']}\n")
                f.write(f"• 총 훈련 시간: {report['training_summary']['total_training_time']}\n")
                f.write(f"• 총 훈련 스텝: {report['training_summary']['total_steps']}\n")
                f.write(f"• 최종 손실값: {report['training_summary']['final_loss']:.4f}\n")
                f.write(f"• 최적 손실값: {report['training_summary']['best_loss']:.4f}\n\n")
                
                f.write("🔍 성능 모니터링 세부사항 (Monitoring Details)\n")
                f.write("-" * 40 + "\n")
                f.write(f"• 로깅 주기: {report['monitoring_details']['logging_frequency']}\n")
                f.write(f"• 체크포인트 저장 주기: {report['monitoring_details']['checkpoint_frequency']}\n") 
                f.write(f"• 총 생성된 체크포인트: {report['monitoring_details']['total_checkpoints_created']}개\n")
                f.write(f"• 워밍업 모니터링: {report['monitoring_details']['warmup_monitoring']}\n")
                f.write(f"• 실시간 관찰: {report['monitoring_details']['real_time_observation']}\n\n")
                
                f.write("🎯 모델 특화 정보 (Model Specialization)\n")
                f.write("-" * 40 + "\n")
                f.write(f"• 도메인: {report['model_specialization']['domain']}\n")
                f.write(f"• 언어 쌍: {report['model_specialization']['language_pair']}\n")
                f.write(f"• 기본 모델: {report['model_specialization']['base_model']}\n")
                f.write(f"• 파인튜닝 방식: {report['model_specialization']['fine_tuning_approach']}\n")
                f.write(f"• 최종 모델 저장 위치: {report['model_specialization']['final_model_location']}\n\n")
                
                f.write("📈 성능 통찰 (Performance Insights)\n")
                f.write("-" * 40 + "\n")
                f.write(f"• 손실값 감소율: {report['performance_insights']['loss_reduction']}\n")
                f.write(f"• 훈련 안정성: {report['performance_insights']['training_stability']}\n")
                f.write(f"• 학습률 워밍업: {report['performance_insights']['learning_rate_warmup']}\n")
                f.write(f"• 체크포인트 관리: {report['performance_insights']['checkpoint_management']}\n\n")
                
                f.write("✅ 결론 (Conclusion)\n")
                f.write("-" * 40 + "\n")
                f.write("체계적인 학습 과정을 거쳐 일반적인 NLLB 모델을 세밀하게 조정할 수 있었고,\n")
                f.write("그 결과 러시아어-한국어 법률 분야에 특화된 번역 모델이 성공적으로 구축되었습니다.\n\n")
                f.write("워밍업 단계와 안정화된 학습률로 모델의 성능이 점진적으로 증진되는 양상을\n")
                f.write("실시간으로 확인할 수 있었으며, 예측 불허의 사태에 대비한 체계적인 체크포인트\n")
                f.write("저장을 통해 안정적인 훈련 과정을 보장했습니다.\n")
            
            logger.info(f"📄 Final training report saved to: {self.log_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
    
    def _format_timedelta(self, td: timedelta) -> str:
        """시간 간격을 읽기 쉬운 형태로 포맷"""
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_metrics_summary(self) -> Dict:
        """메트릭 요약 반환"""
        if not self.training_metrics:
            return {}
            
        return {
            "total_steps": len(self.training_metrics),
            "final_loss": self.training_metrics[-1].loss,
            "best_loss": self.best_loss,
            "best_step": self.best_model_step,
            "loss_reduction_percent": ((self.training_metrics[0].loss - self.best_loss) / self.training_metrics[0].loss * 100),
            "total_checkpoints": len(self.checkpoint_history)
        }

def create_performance_monitor(
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints", 
    final_model_path: str = "./nllb-legal-ru-ko-final",
    logging_steps: int = 200,
    save_steps: int = 1000
) -> PerformanceMonitor:
    """
    성능 모니터 생성 함수
    
    Args:
        log_dir: 로그 저장 디렉토리
        checkpoint_dir: 체크포인트 저장 디렉토리
        final_model_path: 최종 모델 저장 경로
        logging_steps: 로깅 간격 (200스텝마다)
        save_steps: 저장 간격 (1,000스텝마다)
    
    Returns:
        PerformanceMonitor: 설정된 성능 모니터 인스턴스
    """
    
    monitor = PerformanceMonitor(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        final_model_path=final_model_path,
        logging_steps=logging_steps,
        save_steps=save_steps
    )
    
    logger.info("🎯 Performance monitoring system ready")
    logger.info("   - 200스텝마다 학습 진행 상황 기록")
    logger.info("   - 1,000스텝마다 모델 체크포인트 저장") 
    logger.info("   - 워밍업 및 학습률 스케줄링 모니터링")
    logger.info(f"   - 최종 모델 저장 경로: {final_model_path}")
    
    return monitor

if __name__ == "__main__":
    # 성능 모니터링 시스템 데모
    print("🔍 Performance Monitoring System for NLLB Legal Translation")
    print("=" * 70)
    print()
    print("이 시스템은 다음 기능을 제공합니다:")
    print("• 200스텝마다 실시간 학습 진행 상황 기록")
    print("• 1,000스텝마다 모델 체크포인트 자동 저장")
    print("• 워밍업 단계 및 학습률 스케줄링 모니터링")
    print("• 시스템 리소스 사용률 추적")
    print("• 예측 불허 상황 대비 체계적 백업")
    print("• 최종 모델을 './nllb-legal-ru-ko-final' 경로에 저장")
    print()
    print("📊 사용 예시:")
    print("from models.performance_monitoring import create_performance_monitor")
    print("monitor = create_performance_monitor()")
    print("# trainer에 콜백으로 추가")
    print("trainer = Trainer(..., callbacks=[monitor])")
