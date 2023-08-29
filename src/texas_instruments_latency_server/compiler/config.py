from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class TIDebugLevel(IntEnum):
    """
    - NO_DEBUG (0) - no debug,
    - DEBUG_1 (1) - Level 1 debug prints,
    - DEBUG_2 (2) - Level 2 debug prints,
    - DEBUG_3 (3) - Level 1 debug prints, fixed point layer traces,
    - DEBUG_4 (4) (experimental) - Level 1 debug prints, Fixed point and floating point traces,
    - DEBUG_5 (5) (experimental) - Level 2 debug prints, Fixed point and floating point traces
    - DEBUG_6 (6) - Level 3 debug prints

    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md#optional-options
    """

    NO_DEBUG = 0
    DEBUG_1 = 1
    DEBUG_2 = 2
    DEBUG_3 = 3
    DEBUG_4_EXPERIMENTAL = 4
    DEBUG_5_EXPERIMENTAL = 5
    DEBUG_6_EXPERIMENTAL = 6


class TITensorBits(IntEnum):
    """
    - TENSOR_8_BITS (8) - fixed point 8
    - TENSOR_16_BITS (16) - fixed point 16
    - TENSOR_32_BITS (32) - only for PC inference, not device

    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md#optional-options
    """

    TENSOR_8_BITS = 8
    TENSOR_16_BITS = 16
    TENSOR_32_BITS = 32


class TIAccuracyLevel(IntEnum):
    """
    - BASIC (0) - basic calibration,
    - ADVANCED (1) - higher accuracy (advanced bias calibration),
    - USER_DEFINED (9) - user defined

    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md#optional-options
    """

    BASIC = 0
    ADVANCED = 1
    USER_DEFINED = 9


class TIQuantizationScaleType(IntEnum):
    """
    - NON_POWER_OF_2 (0) - non-power-of-2,
    - POWER_OF_2 (1) - power-of-2
    - TFLITE_ASYMMETRIC (3) - asymmetric quantization. Applicable only for pre-quantized tflite models on all SOCs
    except AM68PA.

    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md#advanced-miscellaneous-options
    """

    NON_POWER_OF_2 = 0
    POWER_OF_2 = 1
    TFLITE_ASYMMETRIC = 3


class TIDataConversion(IntEnum):
    """
    - DISABLE (0) - disable
    - INPUT_FORMAT_CONVERSION (1) - Input format conversion
    - OUTPUT_FORMAT_CONVERSION (2) - output format conversion
    - INPUT_OUTPUT_FORMAT_CONVERSION (3) - Input and output format conversion

    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md#advanced-miscellaneous-options
    """

    DISABLE = 0
    INPUT_FORMAT_CONVERSION = 1
    OUTPUT_FORMAT_CONVERSION = 2
    INPUT_OUTPUT_FORMAT_CONVERSION = 3


class _TICfgAsDictMixin:
    @staticmethod
    def _safe_join(value: Optional[List[str]], sep: str = ","):
        return "" if value is None else sep.join(value)

    def as_cfg_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("as_cfg_dict method is not implemented")


@dataclass
class TIModelCfg(_TICfgAsDictMixin):
    """
    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md
    """

    is_od_model: bool = False
    deny_list_layer_type: Optional[List[str]] = None
    deny_list_layer_name: Optional[List[str]] = None
    allow_list_layer_name: Optional[List[str]] = None

    def as_cfg_dict(self):
        return {
            "model_type": "OD" if self.is_od_model else "",
            "deny_list:layer_type": self._safe_join(self.deny_list_layer_type),
            "deny_list:layer_name": self._safe_join(self.deny_list_layer_name),
            "allow_list:layer_name": self._safe_join(self.allow_list_layer_name),
        }


@dataclass
class TIPrecisionCfg(_TICfgAsDictMixin):
    """
    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md
    """

    tensor_bits: TITensorBits
    output_feature_16bit_names_list: Optional[List[str]] = None
    params_16bit_names_list: Optional[List[str]] = None
    mixed_precision_factor: Optional[float] = None

    def as_cfg_dict(self) -> Dict[str, Any]:
        mixed_precision_factor = -1 if self.mixed_precision_factor is None else self.mixed_precision_factor
        return {
            "tensor_bits": self.tensor_bits.value,
            "advanced_options:output_feature_16bit_names_list": self._safe_join(self.output_feature_16bit_names_list),
            "advanced_options:params_16bit_names_list": self._safe_join(self.params_16bit_names_list),
            "advanced_options:mixed_precision_factor": mixed_precision_factor,
        }


@dataclass
class TICalibrationCfg(_TICfgAsDictMixin):
    """
    https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md
    """

    accuracy_level: TIAccuracyLevel
    quantization_scale_type: TIQuantizationScaleType = TIQuantizationScaleType.NON_POWER_OF_2
    high_resolution_optimization: bool = False
    pre_batchnorm_fold: bool = True
    activation_clipping: bool = True
    weight_clipping: bool = True
    bias_calibration: bool = True
    calibration_iterations: int = 5
    add_data_convert_ops: TIDataConversion = TIDataConversion.INPUT_OUTPUT_FORMAT_CONVERSION
    channel_wise_quantization: bool = False

    def as_cfg_dict(self) -> Dict[str, Any]:
        return {
            "accuracy_level": self.accuracy_level.value,
            "advanced_options:quantization_scale_type": self.quantization_scale_type.value,
            "advanced_options:high_resolution_optimization": int(self.high_resolution_optimization),
            "advanced_options:activation_clipping": int(self.activation_clipping),
            "advanced_options:weight_clipping": int(self.weight_clipping),
            "advanced_options:bias_calibration": int(self.bias_calibration),
            "advanced_options:calibration_iterations": self.calibration_iterations,
            "advanced_options:add_data_convert_ops": int(self.add_data_convert_ops),
            "advanced_options:channel_wise_quantization": int(self.channel_wise_quantization),
        }
