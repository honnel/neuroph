package edu.kit.pmk.fantastic4.NeuralNetwork;

public class Connection {

	final Neuron in, out;
	private double weight;

	public Connection(Neuron in, Neuron out) {
		this.in = in;
		this.out = out;
	}

	public double getWeight() {
		return this.weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public void push(double input) {
		out.push(input * weight);
	}

}
