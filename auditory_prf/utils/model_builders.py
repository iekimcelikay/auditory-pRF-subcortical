class ModelBuilderRegistry:
    """Central registry for all model name builders."""

    def __init__(self):
        self._builders = {}

    def register(self, model_name):
        """Decorator to register a model name builder function."""
        def decorator(builder_func):
            self._builders[model_name] = builder_func
            return builder_func
        return decorator

    def get(self, model_name):
        """Retrieve a registered model name builder function."""
        if model_name not in self._builders:
            raise ValueError(f"Unknown model: {model_name}."
                             f"Available: {list(self._builders.keys())}")
        return self._builders[model_name]

    def list_models(self):
        """List all registered model names."""
        return list(self._builders.keys())


# Global registry instance
model_builders = ModelBuilderRegistry()

# Register builders with decarator


@model_builders.register("bez2018")
def bez2018_name_builder(params, timestamp):
    lsr, msr, hsr = params['num_ANF']
    num_runs = params['num_runs']
    num_cf = params['num_cf']
    min_cf = params['min_cf']
    max_cf = params['max_cf']
    return (f"bez2018_psth_batch_{num_runs}runs_{num_cf}cfs_"
            f"{min_cf}-{max_cf}Hz_{lsr}-{msr}-{hsr}fibers_{timestamp}")


@model_builders.register("cochlea_zilany2014")
def cochlea_zilany2014_name_builder(params, timestamp):
    num_runs = params['num_runs']
    num_cf = params['num_cf']
    min_cf = params['min_cf']
    max_cf = params['max_cf']
    return (f"cochlea_zilany2014_psth_batch_"
            f"{num_runs}runs_{num_cf}cfs_{min_cf}-{max_cf}Hz_{timestamp}")


@model_builders.register("wsr_model")
def wsr_model_name_builder(params, timestamp):
    num_channels = params['num_channels']
    frame_length = params['frame_length']
    time_constant = params['time_constant']
    factor = params['factor']
    shift = params['shift']
    return (f"wsr_model_psth_batch_"
            f"{num_channels}chans_{frame_length}ms_"
            f"{time_constant}tc_{factor}factor_{shift}shift_{timestamp}")


@model_builders.register("model_comparison")
def model_comparison_name_builder(params, timestamp):
    """
    Builder for model comparison analyses (e.g., WSR vs BEZ, WSR vs Cochlea).

    Expected params:
        - model1: Name of first model (e.g., 'wsr')
        - model2: Name of second model (e.g., 'bez', 'cochlea')
        - comparison_type: Optional descriptor (e.g., 'population',
                                                'fiber_type')
    """
    model1 = params.get('model1', 'model1')
    model2 = params.get('model2', 'model2')
    comparison_type = params.get('comparison_type', 'comparison')

    return f"{model1}_vs_{model2}_{comparison_type}_{timestamp}"
