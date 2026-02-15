"""End-to-end pipeline integrating all components."""

import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .configs import TrainingConfig
from .data import CoffeeBeanDataPipeline
from .models import ModelFactory
from .training import ModelTrainer
from .evaluation import ClassificationEvaluator, ModelComparator
from .registry import ModelRegistry, ModelMetadata
from .utils import get_logger, ensure_dir, save_json

logger = get_logger(__name__)


class CoffeeBeanPipeline:
    """
    End-to-end pipeline for coffee bean classification.
    
    Integrates all components:
    - Data loading and preprocessing
    - Model creation and training
    - Evaluation and comparison
    - Model registry and versioning
    
    Example:
        >>> pipeline = CoffeeBeanPipeline(config)
        >>> results = pipeline.run()
        >>> print(f"Best model: {results['best_model']}")
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.config.validate()
        
        # Initialize components
        self.data_pipeline = None
        self.trainer = None
        self.registry = None
        
        # Results storage
        self.results = {
            'models_trained': [],
            'evaluations': {},
            'best_model': None,
            'registry_ids': {}
        }
        
        logger.info("CoffeeBeanPipeline initialized")
    
    def run(
        self,
        models_to_train: Optional[List[str]] = None,
        auto_register: bool = True,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            models_to_train: List of models to train (None for all)
            auto_register: Automatically register best models
            save_visualizations: Save evaluation plots
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 70)
        logger.info("STARTING END-TO-END PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Step 1: Setup
            self._setup()
            
            # Step 2: Load and prepare data
            self._prepare_data()
            
            # Step 3: Train models
            training_results = self._train_models(models_to_train)
            
            # Step 4: Evaluate models
            evaluation_results = self._evaluate_models(training_results)
            
            # Step 5: Compare models
            comparison_results = self._compare_models(evaluation_results)
            
            # Step 6: Register best models
            if auto_register:
                self._register_models(evaluation_results, comparison_results)
            
            # Step 7: Generate visualizations
            if save_visualizations:
                self._generate_visualizations(evaluation_results)
            
            # Step 8: Save final report
            self._save_pipeline_report()
            
            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _setup(self) -> None:
        """Setup pipeline components."""
        logger.info("\n[1/7] Setting up pipeline components...")
        
        # Create output directories
        ensure_dir(self.config.get_output_path())
        ensure_dir(self.config.get_output_path('evaluations'))
        ensure_dir(self.config.get_output_path('visualizations'))
        
        # Initialize registry
        registry_path = self.config.get_output_path('registry')
        self.registry = ModelRegistry(str(registry_path))
        
        logger.info("✓ Setup completed")
    
    def _prepare_data(self) -> None:
        """Load and prepare data."""
        logger.info("\n[2/7] Preparing data...")
        
        if self.config.data_config is None:
            raise ValueError("data_config not set in TrainingConfig")
        
        # Initialize data pipeline
        self.data_pipeline = CoffeeBeanDataPipeline(self.config.data_config)
        
        # Load datasets
        train_ds, val_ds, test_ds = self.data_pipeline.load_dataset()
        
        # Get dataset info
        info = self.data_pipeline.get_dataset_info()
        self.results['dataset_info'] = info
        
        logger.info(f"✓ Data prepared: {info['num_classes']} classes")
        logger.info(f"  Train batches: {info.get('train_size', 'N/A')}")
        logger.info(f"  Val batches: {info.get('val_size', 'N/A')}")
        logger.info(f"  Test batches: {info.get('test_size', 'N/A')}")
    
    def _train_models(
        self,
        models_to_train: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Train models."""
        logger.info("\n[3/7] Training models...")
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.config, self.data_pipeline)
        
        # Train models
        if models_to_train is None:
            models_to_train = ['resnet50', 'efficientnet_b0', 'mobilenet_v3']
        
        training_results = self.trainer.train_all_models(
            models_to_train,
            continue_on_error=True
        )
        
        self.results['models_trained'] = list(training_results.keys())
        
        logger.info(f"✓ Trained {len(training_results)} models")
        return training_results
    
    def _evaluate_models(
        self,
        training_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models."""
        logger.info("\n[4/7] Evaluating models...")
        
        _, _, test_ds = self.data_pipeline.get_datasets()
        class_names = self.data_pipeline.get_class_names()
        
        evaluation_results = {}
        
        for model_name in self.results['models_trained']:
            logger.info(f"  Evaluating {model_name}...")
            
            # Load best model from checkpoint
            checkpoint_path = self.config.get_output_path('checkpoints') / f"{model_name}_best.h5"
            
            if checkpoint_path.exists():
                model = tf.keras.models.load_model(checkpoint_path)
                
                # Create evaluator
                evaluator = ClassificationEvaluator(
                    model,
                    class_names,
                    output_dir=str(self.config.get_output_path('evaluations') / model_name)
                )
                
                # Evaluate
                metrics = evaluator.evaluate(test_ds, verbose=0)
                
                # Store results
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'evaluator': evaluator
                }
                
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            else:
                logger.warning(f"    Checkpoint not found for {model_name}")
        
        self.results['evaluations'] = {
            k: v['metrics'] for k, v in evaluation_results.items()
        }
        
        logger.info(f"✓ Evaluated {len(evaluation_results)} models")
        return evaluation_results
    
    def _compare_models(
        self,
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare all models."""
        logger.info("\n[5/7] Comparing models...")
        
        # Create comparator
        comparator = ModelComparator()
        
        for model_name, results in evaluation_results.items():
            comparator.add_model(model_name, results['metrics'])
        
        # Get comparison
        comparison_df = comparator.get_comparison_df()
        
        # Get best model
        best_model, best_value = comparator.get_best_model('f1_macro')
        
        self.results['best_model'] = best_model
        self.results['best_f1_score'] = best_value
        self.results['comparison'] = comparison_df.to_dict()
        
        logger.info(f"✓ Best model: {best_model} (F1={best_value:.4f})")
        
        # Save comparison plot
        viz_path = self.config.get_output_path('visualizations')
        comparator.plot_comparison(save_path=str(viz_path / 'model_comparison.png'))
        
        return {
            'comparator': comparator,
            'best_model': best_model,
            'best_value': best_value
        }
    
    def _register_models(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        comparison_results: Dict[str, Any]
    ) -> None:
        """Register models in registry."""
        logger.info("\n[6/7] Registering models...")
        
        for model_name, results in evaluation_results.items():
            # Load model
            checkpoint_path = self.config.get_output_path('checkpoints') / f"{model_name}_best.h5"
            
            if checkpoint_path.exists():
                model = tf.keras.models.load_model(checkpoint_path)
                
                # Create metadata
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                tags = ['trained']
                if model_name == comparison_results['best_model']:
                    tags.append('best')
                    tags.append('production')
                
                metadata = ModelMetadata(
                    model_name=model_name,
                    version=version,
                    architecture=model_name,
                    created_at=datetime.now().isoformat(),
                    metrics={
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in results['metrics'].items()
                        if isinstance(v, (int, float))
                    },
                    config=self.config.to_dict(),
                    input_shape=self.config.model_config.input_shape if self.config.model_config else (224, 224, 3),
                    num_classes=self.data_pipeline.get_num_classes(),
                    tags=tags,
                    description=f"Trained on {datetime.now().date()}"
                )
                
                # Register
                model_id = self.registry.register_model(model, metadata)
                self.results['registry_ids'][model_name] = model_id
                
                logger.info(f"  Registered {model_id}")
        
        logger.info(f"✓ Registered {len(self.results['registry_ids'])} models")
    
    def _generate_visualizations(
        self,
        evaluation_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Generate and save visualizations."""
        logger.info("\n[7/7] Generating visualizations...")
        
        viz_path = self.config.get_output_path('visualizations')
        
        for model_name, results in evaluation_results.items():
            evaluator = results['evaluator']
            model_viz_path = viz_path / model_name
            ensure_dir(model_viz_path)
            
            try:
                # Confusion matrix
                evaluator.plot_confusion_matrix(
                    save_path=str(model_viz_path / 'confusion_matrix.png')
                )
                
                # Per-class metrics
                evaluator.plot_per_class_metrics(
                    save_path=str(model_viz_path / 'per_class_metrics.png')
                )
                
                # ROC curves
                evaluator.plot_roc_curves(
                    save_path=str(model_viz_path / 'roc_curves.png')
                )
                
                # Generate report
                evaluator.generate_report(
                    save_path=str(model_viz_path / 'evaluation_report.txt')
                )
                
                logger.info(f"  Generated visualizations for {model_name}")
                
            except Exception as e:
                logger.warning(f"  Failed to generate some visualizations for {model_name}: {e}")
        
        logger.info("✓ Visualizations generated")
    
    def _save_pipeline_report(self) -> None:
        """Save final pipeline report."""
        report_path = self.config.get_output_path() / 'pipeline_report.json'
        
        # Clean results for JSON serialization
        clean_results = {
            'pipeline_completed': datetime.now().isoformat(),
            'config': {
                'experiment_name': self.config.experiment_name,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
            },
            'dataset_info': self.results.get('dataset_info', {}),
            'models_trained': self.results['models_trained'],
            'best_model': self.results.get('best_model'),
            'best_f1_score': self.results.get('best_f1_score'),
            'registry_ids': self.results.get('registry_ids', {}),
        }
        
        save_json(clean_results, report_path)
        logger.info(f"\n✓ Pipeline report saved to {report_path}")
    
    def get_best_model(self) -> tf.keras.Model:
        """
        Get the best trained model.
        
        Returns:
            Best Keras model
        """
        if not self.results.get('best_model'):
            raise ValueError("Pipeline not completed yet")
        
        best_model_name = self.results['best_model']
        checkpoint_path = self.config.get_output_path('checkpoints') / f"{best_model_name}_best.h5"
        
        return tf.keras.models.load_model(checkpoint_path)