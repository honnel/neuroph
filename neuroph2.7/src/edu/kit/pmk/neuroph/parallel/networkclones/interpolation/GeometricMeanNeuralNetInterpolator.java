package edu.kit.pmk.neuroph.parallel.networkclones.interpolation;

import org.neuroph.core.Weight;

public class GeometricMeanNeuralNetInterpolator extends NeuralNetInterpolator {

	@Override
	public WeightAdjustment getWeightAdjustmentForIndex(
			Weight[][] neuronWeights, int index) {
		WeightAdjustment geom = new WeightAdjustment();
		geom.adjustedWeight = 0;
		geom.adjustedWeightChange = 0;
		for (int net = 0; net < neuronWeights.length; net++) {
			geom.adjustedWeight += Math.log(neuronWeights[net][index].value);
			geom.adjustedWeightChange += Math
					.log(neuronWeights[net][index].weightChange);
		}
		geom.adjustedWeight = Math.exp(geom.adjustedWeight
				/ neuronWeights.length);
		geom.adjustedWeightChange = Math.exp(geom.adjustedWeightChange
				/ neuronWeights.length);
		return geom;
	}

}
