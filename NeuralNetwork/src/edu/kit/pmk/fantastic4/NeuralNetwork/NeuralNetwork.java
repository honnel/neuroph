package edu.kit.pmk.fantastic4.NeuralNetwork;

public abstract class NeuralNetwork {

	public static NeuralNetwork createNeuralNetwork(NeuralNetworkType type) {
		switch (type) {
		case Layered:
			return new LayeredNeuralNetwork();
		default:
			return null;
		}
	}

}
