package edu.kit.pmk.neuroph.parallel.networkclones.interpolation;

import org.neuroph.core.Weight;

public class MaximumNeuralNetInterpolator extends NeuralNetInterpolator {

	@Override
	public WeightAdjustment getWeightAdjustmentForIndex(
			Weight[][] neuronWeights, int index) {
		WeightAdjustment max = new WeightAdjustment();
		max.adjustedWeight = Double.MIN_VALUE;
		max.adjustedWeightChange = Double.MIN_VALUE;
		for (int net = 0; net < neuronWeights.length; net++) {
			max.adjustedWeight = Math.max(max.adjustedWeight,
					neuronWeights[net][index].value);
			max.adjustedWeightChange = Math.max(max.adjustedWeightChange,
					neuronWeights[net][index].weightChange);
		}
		return max;
	}

}
