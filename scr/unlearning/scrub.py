import utils

def unlearn(trainer, config):
    for epoch in range(config.num_total_epochs):
        trainer.loss_type = 'scrub_maximize'
        trainer.train()
        utils.clear_cache()

        trainer.loss_type = 'scrub_minimize'
        trainer.train()
        utils.clear_cache()
