import logging
import argparse
import os

from .config import load_config
from .cfr_trainer import CFRTrainer

def setup_logging(config):
    """Configures logging based on the loaded configuration."""
    level = getattr(logging, config.logging.log_level.upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handlers = []
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    handlers.append(ch)

    # File Handler (Optional)
    if config.logging.log_file:
        try:
            log_dir = os.path.dirname(config.logging.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(config.logging.log_file, mode='a') # Append mode
            fh.setLevel(level)
            fh.setFormatter(formatter)
            handlers.append(fh)
        except Exception as e:
            print(f"Warning: Could not set up file logging at {config.logging.log_file}: {e}")

    logging.basicConfig(level=level, handlers=handlers)
    logging.getLogger("asyncio").setLevel(logging.WARNING) # Reduce verbosity from libraries


def main():
    parser = argparse.ArgumentParser(description="Run CFR+ Training for Cambia")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (overrides config if provided)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing agent data before training",
    )
    parser.add_argument(
         "--save_path",
         type=str,
         default=None,
         help="Override save path for agent data (from config)"
    )


    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Cambia CFR+ Training ---")
    logger.info(f"Configuration loaded from: {args.config}")

    # Override config settings from command line if provided
    if args.iterations is not None:
        config.cfr_training.num_iterations = args.iterations
        logger.info(f"Overriding iterations from command line: {args.iterations}")
    if args.save_path is not None:
         config.persistence.agent_data_save_path = args.save_path
         logger.info(f"Overriding save path from command line: {args.save_path}")


    # Initialize trainer
    trainer = CFRTrainer(config)

    # Load existing data if requested
    if args.load:
        logger.info(f"Attempting to load agent data from: {config.persistence.agent_data_save_path}")
        trainer.load_data()
    else:
        logger.info("Starting training from scratch (no data loaded).")

    # Run training
    try:
        trainer.train() # Uses iterations from config (potentially overridden)
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Attempting to save current progress...")
        trainer.save_data()
        logger.info("Progress saved.")
    except Exception as e:
        logger.exception("An unexpected error occurred during training:") # Logs traceback
        logger.info("Attempting to save current progress before exiting...")
        try:
            trainer.save_data()
            logger.info("Progress saved.")
        except Exception as save_e:
            logger.error(f"Failed to save progress after error: {save_e}")


if __name__ == "__main__":
    main()