from apscheduler.schedulers.blocking import BlockingScheduler
from ml.train import train_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scheduler = BlockingScheduler()

@scheduler.scheduled_job("cron", day_of_week="sat", hour=12, minute=0)
def scheduled_training():
    logger.info("Saturday 12:00 noon — Starting scheduled retraining...")
    try:
        results = train_models()
        logger.info(f"Retraining complete! Results: {results}")
    except Exception as e:
        logger.error(f" Retraining failed: {e}")

if __name__ == "__main__":
    logger.info(" Scheduler started — will retrain every Saturday at 12:00 noon")
    scheduler.start()