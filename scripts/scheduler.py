from apscheduler.schedulers.blocking import BlockingScheduler
from ml.train import train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler()

@scheduler.scheduled_job("cron", day_of_week="sat", hour=12, minute=0)
def retrain():
    logger.info("⏰ Saturday 12:00 — Retraining model...")
    try:
        metrics = train_model()
        logger.info(f"✅ Retraining done! Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")

if __name__ == "__main__":
    logger.info("⏰ Scheduler running — retrains every Saturday at noon")
    scheduler.start()