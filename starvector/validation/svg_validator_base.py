from abc import ABC, abstractmethod
import os
import json
import pandas as pd
from starvector.metrics.metrics import SVGMetrics
from copy import deepcopy
import numpy as np
from starvector.data.util import rasterize_svg
import importlib
from typing import Type
from omegaconf import OmegaConf
from tqdm import tqdm   
from datetime import datetime
import re
from starvector.data.util import clean_svg, use_placeholder
from svgpathtools import svgstr2paths

# Registry for SVGValidator subclasses
validator_registry = {}

def register_validator(cls: Type['SVGValidator']):
    """
    Decorator to register SVGValidator subclasses.
    """
    validator_registry[cls.__name__] = cls
    return cls

class SVGValidator(ABC):
    def __init__(self, config):
        self.task = config.model.task
        # Flag to determine if we should report to wandb
        self.report_to_wandb = config.run.report_to == 'wandb'
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.model.from_checkpoint:
            chkp_dir = self.get_checkpoint_dir(config.model.from_checkpoint)
            config.model.from_checkpoint = chkp_dir
            self.resume_from_checkpoint = chkp_dir
            self.out_dir = chkp_dir + '/' + config.run.out_dir + '/' + config.model.generation_engine + '_' + config.dataset.dataset_name + '_' + date_time
        else:
            self.out_dir = config.run.out_dir + '/' + config.model.generation_engine + '_' + config.model.name + '_' + config.dataset.dataset_name + '_' + date_time
        os.makedirs(self.out_dir, exist_ok=True)
        self.model_name = config.model.name
        # Save config to yaml file
        config_path = os.path.join(self.out_dir, "config.yaml")
        self.config = config
        with open(config_path, "w") as f:
            OmegaConf.save(config=self.config, f=f)

        print(f"Out dir: {self.out_dir}")
        os.makedirs(self.out_dir, exist_ok=True)
        
        metrics_config_path = f"configs/metrics/{self.task}.yaml"
        default_metrics_config = OmegaConf.load(metrics_config_path)
        self.metrics = SVGMetrics(default_metrics_config['metrics'])
        self.results = {}

        # If wandb reporting is enabled, initialize wandb and a table to record sample results.
        if self.report_to_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.run.project_name,
                    name=config.run.run_id,
                    config=OmegaConf.to_container(config, resolve=True)
                )
                # Create a wandb table with columns for all relevant data.
                self.results_table = wandb.Table(columns=[
                    "sample_id", "svg", "svg_raw", "svg_gt",
                    "no_compile", "post_processed", "original_image", "generated_image",
                    "comparison_image"
                ])
                # Dictionary to hold table rows indexed by sample_id
                self.table_data = {}
                print("Initialized wandb run with results table")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")

    def get_checkpoint_dir(self, checkpoint_path):
        """Get the directory of a checkpoint by name, returning the one with the highest step."""

        if re.search(r'checkpoint-\d+$', checkpoint_path):
            return checkpoint_path

        # Find all directories matching the checkpoint pattern
        checkpoint_dirs = []
        for d in os.listdir(checkpoint_path):
            if re.search(r'checkpoint-(\d+)$', d):
                checkpoint_dirs.append(d)
        
        if not checkpoint_dirs:
            return None
            
        # Extract step numbers and find the highest one
        latest_dir = max(checkpoint_dirs, key=lambda x: int(re.search(r'checkpoint-(\d+)$', x).group(1)))
        return os.path.join(checkpoint_path, latest_dir)

    def _hash_config(self, config):
        """Create a deterministic hash of the config for caching/identification."""
        import json
        import hashlib
        
        # Convert OmegaConf to dict and sort it for deterministic serialization
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # Remove non-deterministic or irrelevant fields
        if 'run' in config_dict:
            config_dict['run'].pop('out_dir', None)  # Remove output directory
            config_dict['run'].pop('device', None)   # Remove device specification
        
        # Convert to sorted JSON string
        config_str = json.dumps(config_dict, sort_keys=True)
        
        # Create hash
        return hashlib.md5(config_str.encode()).hexdigest()

    @abstractmethod
    def generate_svg(self, batch):
        """Generate SVG from batch data"""
        pass
    
    @abstractmethod
    def post_process_svg(self, generated_output):
        """Post-process generated SVG"""
        pass


    def create_comparison_plot(self, sample_id, gt_raster, gen_raster, metrics, output_path):
        """
        Creates and saves a comparison plot showing the ground truth and generated SVG images, along with computed metrics.
        
        Args:
            sample_id (str): Identifier for the sample.
            gt_raster (PIL.Image.Image): Rasterized ground truth SVG image.
            gen_raster (PIL.Image.Image): Rasterized generated SVG image.
            metrics (dict): Dictionary of metric names and their values.
            output_path (str): File path where the plot is saved.
            
        Returns:
            PIL.Image.Image: The generated comparison plot image.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image

        # Create figure with two subplots: one for metrics text, one for the images
        fig, (ax_metrics, ax_images) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 4]})
        fig.suptitle(f'Generation Results for {sample_id}', fontsize=16)

        # Build text for metrics
        if metrics:
            metrics_text = "Metrics:\n"
            for key, val in metrics.items():
                if isinstance(val, list) and val:
                    metrics_text += f"{key}: {val[-1]:.4f}\n"
                elif isinstance(val, (int, float)):
                    metrics_text += f"{key}: {val:.4f}\n"
                else:
                    metrics_text += f"{key}: {val}\n"
        else:
            metrics_text = "No metrics available."
        
        # Add metrics text in the upper subplot
        ax_metrics.text(0.5, 0.5, metrics_text, fontfamily='monospace',
                        horizontalalignment='center', verticalalignment='center')
        ax_metrics.axis('off')

        # Set title and prepare the images subplot
        ax_images.set_title('Ground Truth (left) vs Generated (right)')
        gt_array = np.array(gt_raster)
        gen_array = np.array(gen_raster)
        combined = np.hstack((gt_array, gen_array))
        ax_images.imshow(combined)
        ax_images.axis('off')

        # Save figure to buffer and file path
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    def create_comparison_plots_with_metrics(self, all_metrics):
        """
        Create and save comparison plots with metrics for all samples based on computed metrics.
        """
        for sample_id, metrics in all_metrics.items():
            if sample_id not in self.results:
                continue  # Skip if the sample does not exist in the results
            
            result = self.results[sample_id]
            sample_dir = os.path.join(self.out_dir, sample_id)
            
            # Retrieve the already rasterized images from the result
            gt_raster = result.get('gt_im')
            gen_raster = result.get('gen_im')
            if gt_raster is None or gen_raster is None:
                continue
            
            # Define the output path for the comparison plot image
            output_path = os.path.join(sample_dir, f"{sample_id}_comparison.png")
            comp_img = self.create_comparison_plot(sample_id, gt_raster, gen_raster, metrics, output_path)
            
            # Save the generated plot image in the result for later use
            result['comparison_image'] = comp_img

            # Also update the row in the internal table_data with the comparison image.
            if self.report_to_wandb and sample_id in self.table_data and self.config.run.log_images:
                import wandb
                row = list(self.table_data[sample_id])
                row[-1] = wandb.Image(comp_img)
                self.table_data[sample_id] = tuple(row)
                self.update_results_table_log()

    def save_results(self, results, batch, batch_idx):
        """Save results from generation."""
        out_path = self.out_dir
        for i, sample in enumerate(batch['Svg']):
            sample_id = str(batch['Filename'][i]).split('.')[0]
            res = results[i]
            res['sample_id'] = sample_id
            res['gt_svg'] = sample
            
            sample_dir = os.path.join(out_path, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save SVG files and rasterized images using the base class method
            svg_raster, gt_svg_raster = self._save_svg_files(sample_dir, sample_id, res)
            
            # Save metadata to disk
            with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
                json.dump(res, f, indent=4, sort_keys=True)
            res['gen_im'] = svg_raster
            res['gt_im'] = gt_svg_raster

            self.results[sample_id] = res

            # Instead of logging individual sample fields directly, add an entry (row)
            # to the internal table_data with a placeholder for comparison_image.
            if self.report_to_wandb and self.config.run.log_images:
                import wandb
                row = (
                    sample_id,
                    res['svg'],
                    res['svg_raw'],
                    res['gt_svg'],
                    res['no_compile'],
                    res['post_processed'],
                    wandb.Image(gt_svg_raster),
                    wandb.Image(svg_raster),
                    None  # Placeholder for comparison_image
                )
                self.table_data[sample_id] = row
                self.update_results_table_log()
    
    def _save_svg_files(self, sample_dir, outpath_filename, res):
        """Save SVG files and rasterized images."""
        # Save SVG files
        with open(os.path.join(sample_dir, f"{outpath_filename}.svg"), 'w', encoding='utf-8') as f:
            f.write(res['svg'])
        with open(os.path.join(sample_dir, f"{outpath_filename}_raw.svg"), 'w', encoding='utf-8') as f:
            f.write(res['svg_raw'])
        with open(os.path.join(sample_dir, f"{outpath_filename}_gt.svg"), 'w', encoding='utf-8') as f:
            f.write(res['gt_svg'])
        
        # Rasterize and save PNG
        svg_raster = rasterize_svg(res['svg'], resolution=512, dpi=100, scale=1)
        gt_svg_raster = rasterize_svg(res['gt_svg'], resolution=512, dpi=100, scale=1)
        svg_raster.save(os.path.join(sample_dir, f"{outpath_filename}_generated.png"))
        gt_svg_raster.save(os.path.join(sample_dir, f"{outpath_filename}_original.png"))
        
        return svg_raster, gt_svg_raster
    
    def run_temperature_sweep(self, batch):
        """Run generation with different temperatures"""
        out_dict = {}
        sampling_temperatures = np.linspace(
            self.config.generation_sweep.min_temperature, 
            self.config.generation_sweep.max_temperature, 
            self.config.generation_sweep.num_generations_different_temp
        ).tolist()
        
        for temp in sampling_temperatures:
            current_args = deepcopy(self.config.generation_params)
            current_args['temperature'] = temp
            results = self.generate_and_process_batch(batch, current_args)
            
            for i, sample_id in enumerate(batch['id']):
                sample_id = str(sample_id).split('.')[0]
                if sample_id not in out_dict:
                    out_dict[sample_id] = {}
                out_dict[sample_id][temp] = results[i]
        
        return out_dict
    
    def validate(self):
        """Main validation loop"""        
        for i, batch in enumerate(tqdm(self.dataloader, desc="Validating")):
            if self.config.generation_params.generation_sweep:
                results = self.run_temperature_sweep(batch)
            else:
                results = self.generate_and_process_batch(batch, self.config.generation_params)

            self.save_results(results, batch, i)

        self.release_memory()
        
        # Calculate and save metrics
        self.calculate_and_save_metrics()

        # Final logging of the complete results table.
        if self.report_to_wandb and self.config.run.log_images:
            try:
                import wandb
                wandb.log({"results_table": self.results_table})
            except Exception as e:
                print(f"Failed to log final results table to wandb: {e}")
    
    def calculate_and_save_metrics(self):
        """Calculate and save metrics"""
        batch_results = self.preprocess_results()
        avg_results, all_results = self.metrics.calculate_metrics(batch_results)
        
        out_path_results = os.path.join(self.out_dir, 'results')
        os.makedirs(out_path_results, exist_ok=True)
        
        # Save average results
        with open(os.path.join(out_path_results, 'results_avg.json'), 'w') as f:
            json.dump(avg_results, f, indent=4, sort_keys=True)        
        # Save detailed results
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df.to_csv(os.path.join(out_path_results, 'all_results.csv'))
        
        # Log average metrics to wandb if enabled
        if self.report_to_wandb:
            try:
                import wandb
                wandb.log({'avg_metrics': avg_results})
            except Exception as e:
                print(f"Error logging average metrics to wandb: {e}")
        
        # Create comparison plots with metrics
        self.create_comparison_plots_with_metrics(all_results)
    
    def preprocess_results(self):
        """Preprocess results from self.results into batch format with lists"""
        batch = {
            'gen_svg': [],
            'gt_svg': [],
            'gen_im': [],
            'gt_im': [],
            'json': []
        }

        for sample_id, result_dict in self.results.items():
            # For single temperature case, result_dict contains one result
            # For temperature sweep, take first temperature's result
            if self.config.generation_params.generation_sweep:
                result = result_dict[list(result_dict.keys())[0]]
            else:
                result = result_dict

            batch['gen_svg'].append(result['svg'])
            batch['gt_svg'].append(result['gt_svg'])
            batch['gen_im'].append(result['gen_im'])
            batch['gt_im'].append(result['gt_im'])
            batch['json'].append(result)

        return batch
    
    def generate_and_process_batch(self, batch, generate_config):
        """Generate and post-process SVGs for a batch"""
        generated_outputs = self.generate_svg(batch, generate_config)
        processed_results = [self.post_process_svg(output) for output in generated_outputs]
        return processed_results


    def post_process_svg(self, text):
        """Post-process a single SVG text"""
        try:
            svgstr2paths(text)
            return {
                'svg': text,
                'svg_raw': text,
                'post_processed': False,
                'no_compile': False
            }
        except:
            try:
                cleaned_svg = clean_svg(text)
                svgstr2paths(cleaned_svg)
                return {
                    'svg': cleaned_svg,
                    'svg_raw': text,
                    'post_processed': True,
                    'no_compile': False
                }
            except:
                return {
                    'svg': use_placeholder(),
                    'svg_raw': text,
                    'post_processed': True,
                    'no_compile': True
                }
    

    @classmethod
    def get_validator(cls, key, args, validator_configs):
        """
        Factory method to get the appropriate SVGValidator subclass based on the key.

        Args:
            key (str): The key name to select the validator.
            args (argparse.Namespace): Parsed command-line arguments.
            validator_configs (dict): Mapping of validator keys to class paths.

        Returns:
            SVGValidator: An instance of a subclass of SVGValidator.

        Raises:
            ValueError: If the provided key is not in the mapping.
        """
        if key not in validator_configs:
            available_validators = list(validator_configs.keys())
            raise ValueError(f"Validator '{key}' is not recognized. Available validators: {available_validators}")
        
        class_path = validator_configs[key]
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        validator_class = getattr(module, class_name)
        
        return validator_class(args)
    
    def update_results_table_log(self):
        """Rebuild and log the results table from self.table_data."""
        if self.report_to_wandb and self.config.run.log_images:
            try:
                import wandb
                table = wandb.Table(columns=[
                    "sample_id", "svg", "svg_raw", "svg_gt",
                    "no_compile", "post_processed",
                    "original_image", "generated_image", "comparison_image"
                ])
                for row in self.table_data.values():
                    table.add_data(*row)
                wandb.log({"results_table": table})
                self.results_table = table
            except Exception as e:
                print(f"Failed to update results table to wandb: {e}")
    