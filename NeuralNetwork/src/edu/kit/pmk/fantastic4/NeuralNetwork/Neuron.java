package edu.kit.pmk.fantastic4.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

import edu.kit.pmk.fantastic4.NeuralNetwork.TransferFunction.TransferFunction;

public class Neuron {

	final TransferFunction func;
	private final List<Double> weightedInputs;
	private final List<Connection> connections;

	public Neuron(TransferFunction func) {
		this.func = func;
		this.weightedInputs = new ArrayList<Double>();
		this.connections = new ArrayList<Connection>();
	}

	// only called once when instantiated
	public void setConnections(List<Connection> connections) {
		this.connections.addAll(connections);
	}

	public synchronized void push(double weightedInput) {
		weightedInputs.add(weightedInput);
		if (weightedInputs.size() == connections.size()) {
			flush();
		}
	}

	private double flush() {
		double output = func.getOutput(weightedInputs);
		for (Connection con : connections) {
			con.push(output);
		}
		return output;
	}

}
