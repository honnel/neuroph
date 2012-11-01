package edu.kit.pmk.fantastic4.NeuralNetwork.TransferFunction;

import java.util.Collection;

public class Threshold implements TransferFunction {

	private final double threshold;

	public Threshold(double threshold) {
		this.threshold = threshold;
	}

	@Override
	public double getOutput(Collection<Double> inputs) {
		double sum = 0;
		for (double in : inputs) {
			sum += in;
		}
		if (sum >= threshold)
			return 1;
		else
			return 0;
	}

}
