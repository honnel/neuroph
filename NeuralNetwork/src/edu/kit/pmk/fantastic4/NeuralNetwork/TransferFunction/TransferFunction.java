package edu.kit.pmk.fantastic4.NeuralNetwork.TransferFunction;

import java.util.Collection;

public interface TransferFunction {
	
	double getOutput(Collection<Double> inputs);

}
