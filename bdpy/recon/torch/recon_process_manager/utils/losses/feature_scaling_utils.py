#coding:utf-8
import warnings

import numpy as np
import copy

def is_in_and_not_None(target_dict, key):
    return key in target_dict and target_dict[key] is not None

def factor_settings_dict_to_factor(original_feature, current_feature, factor_settings: dict, layer: str):
    if 'target_layer' not in factor_settings or layer in factor_settings['target_layers']:
        if isinstance(factor_settings['values'], str):
            func_name = factor_settings['values']
            assert func_name in ['mean', 'std', 'original_mean', 'original_std']
            if 'original' in func_name:
                target_feature = original_feature
            else:
                target_feature = current_feature
            assert 'calculation_type' in factor_settings
            calculation_type = factor_settings['calculation_type']
            if calculation_type == 'layer-wise' or calculation_type == 'all_units_to_one':
                if calculation_type == 'all_units_to_one':
                    warnings.warn("the calcualtion type 'all_units_to_one' is deprecated and will be deleted")
                axis = None
            elif calculation_type == 'channel-wise' and target_feature.ndim >= 3: # expect BxCx(other dimensions); only channel-first array can be handled
                axis_list = list(np.arange(target_feature.ndim))
                axis_list.remove(1)
                axis = tuple(axis_list)
            elif calculation_type == 'position-wise' or calculation_type == 'positional' and target_feature.ndim == 4: # expect BxCxHxW; array with 5 or more dimensions, or 3-dimensional 1 channel array will not be handled
                if calculation_type == 'positional':
                    warnings.warn("the calculation type 'positional' is deprecated and will be deleted")
                axis = (0, 1)
                # # if original_feature.ndim == 4:
                # if target_feature.ndim == 4: # expect BxCxHxW; array with 5 or more dimensions, or 3-dimensional 1 channel array will not be handled
                #     axis = (0, 1)
                # else:
                #     axis = None
            else:
                axis = None # since the decoded feature is a single element batch, unit-wise calculation cannot be performed. thus not `axis=0`

            if func_name == 'mean':
                return np.mean(target_feature, axis=axis, keepdims=True)
            else:
                if 'std_ddof' in factor_settings:
                    std_ddof = factor_settings['std_ddof']
                else:
                    print('std_ddof is set to 0')
                    std_ddof = 0
                return np.std(target_feature, axis=axis, ddof=std_ddof, keepdims=True)
        else:
            return factor_settings['values'][layer]
    return None

def prepare_normalization_factors(original_feature, current_feature, factor_settings, layer: str):
    if isinstance(factor_settings, dict):
        factor = factor_settings_dict_to_factor(original_feature, current_feature, factor_settings, layer)
    else:
        assert isinstance(factor_settings, list)
        factor = None
        for factor_settings_i in factor_settings:
            if factor is not None:
                break
            factor = factor_settings_dict_to_factor(original_feature, current_feature, factor_settings_i, layer)
    return factor

def normalize_features(target_features: dict,
                       normalization_settings: dict):
    '''
    returns (features - subtracthend) / divisor * multiplier + addend for each layer
    normalization_settings:
        {subtrahend: {'values': <ndarray>, 'calculation_type': 'unit-wise'},
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: {'values': <ndarray>, 'calculation_type': 'positional', 'std_ddof': 0},
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - If values is str (choices are ['std', 'mean']), values calculated from original given features will be used
        {subtrahend: {'values': 'mean', 'calculation_type': 'unit-wise'},
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: {'values': 'std', 'calculation_type': 'positional', 'std_ddof': 1},
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - None is acceptable for all elements
        {subtrahend: None,
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: None,
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - If 'target_layers' in the sub-dict, they will applied to only layers in the list.
        {subtrahend: {'values': <ndarray>, 'calculation_type': 'unit-wise', 'target_layers': <list of layers>},
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: {'values': <ndarray>, 'calculation_type': 'positional', 'std_ddof': 0},
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - If you want to use different values for some layers, use following:
        {subtrahend: [{'values': <ndarray>, 'target_layers': <list of layers>, 'calculation_type': 'unit-wise'},
                        {'values': <ndarray>, 'target_layers': <list of layers>, 'calculation_type': 'positional'}],
         divisor: None
         multiplier: None
         addend: None}
    '''

    for layer, feature in target_features.items():
        print(layer, end=':\n')
        original_feature = copy.deepcopy(feature)
        if is_in_and_not_None(normalization_settings, 'subtrahend') or is_in_and_not_None(normalization_settings, 'subtracthend'):
            if is_in_and_not_None(normalization_settings, 'subtracthend'):
                warning.warn('the word "subtracthend" is mistyping and will be changed to "subtrahend"')
            if is_in_and_not_None(normalization_settings, 'subtrahend'):
                subtrahend = prepare_normalization_factors(original_feature, feature, normalization_settings['subtrahend'], layer)
            else:
                subtrahend = prepare_normalization_factors(original_feature, feature, normalization_settings['subtracthend'], layer)
            if subtrahend is not None:
                feature = feature - subtrahend
                print('subtrahend:', subtrahend.shape, np.mean(subtrahend))
            else:
                print('subtrahend: None')

        if is_in_and_not_None(normalization_settings, 'divisor') or is_in_and_not_None(normalization_settings, 'dividor'):
            if is_in_and_not_None(normalization_settings, 'dividor'):
                warning.warn('the word "dividor" is mistyping and will be changed to "divisor"')
            if is_in_and_not_None(normalization_settings, 'divisor'):
                divisor = prepare_normalization_factors(original_feature, feature, normalization_settings['divisor'], layer)
            else:
                divisor = prepare_normalization_factors(original_feature, feature, normalization_settings['dividor'], layer)
            if divisor is not None:
                if np.any(divisor == 0):
                    warnings.warn('there are {} 0 values in the dividor'.format(np.sum(divisor == 0)))
                    dividor = np.where(divisor == 0, 1, divisor)
                feature = feature / divisor
                print('divisor:', divisor.shape, np.mean(divisor))
            else:
                print('divisor: None')

        if normalization_settings['multiplier'] is not None:
            multiplier = prepare_normalization_factors(original_feature, feature, normalization_settings['multiplier'], layer)
            if multiplier is not None:
                if np.any(multiplier == 0):
                    warnings.warn('there are {} 0 values in the multiplier'.format(np.sum(multiplier == 0)))
                feature = feature * multiplier
                print('multiplier:', multiplier.shape, np.mean(multiplier))
            else:
                print('multiplier: None')

        if normalization_settings['addend'] is not None:
            addend = prepare_normalization_factors(original_feature, feature, normalization_settings['addend'], layer)
            if addend is not None:
                feature = feature + addend
                print('addend:', addend.shape, np.mean(addend))
            else:
                print('addend: None')

        target_features[layer] = feature

    return target_features
