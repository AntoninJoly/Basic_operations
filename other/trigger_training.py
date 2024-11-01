from clearml import Dataset
from clearml.automation import TriggerScheduler

if __name__ == "__main__":
    trigger = TriggerScheduler(pooling_frequency_minutes=1.0)
    trigger.add_dataset_trigger(name="Titanic test trigerring",
                                schedule_task_id="bd32f0ba12bc46dea7e0c82e109d08b2",
                                trigger_project="Titanic test",
                                trigger_name="Test dataset",
                                trigger_on_tags=["for trigger"],
                                schedule_queue="cpu",
                                target_project="Titanic test/trigger",)
    trigger.start()