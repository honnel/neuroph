package edu.kit.pmk.fantastic4.NeuralNetwork.TransferFunction;

import java.util.Collection;

public class Sigmoid implements TransferFunction {

	public Sigmoid() {
	}

	@Override
	public double getOutput(Collection<Double> inputs) {
		double sum = 0;
		for (double in : inputs) {
			sum += in;
		}
		return Math.pow(Math.E, sum);
	}

}
