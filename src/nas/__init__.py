"""Neural Architecture Search for the byte-CNN news encoder.

Three arms share one search space but differ in precision + constraints:
  * nas                  -> FP32,   loose (laptop) constraints  -> accuracy ceiling
  * micro_nas            -> INT8,   hard STM32H7 flash/RAM/MAC constraints
  * binarized_micro_nas  -> Binary, hard STM32H7 constraints

Fitness = quick distillation quality (cosine to teacher anchors) under the arm's
precision (QAT), subject to footprint feasibility. This makes architecture the
controlled axis and precision the second axis of the results matrix.
"""
