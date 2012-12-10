package edu.kit.pmk.neuroph.parallel;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;

public interface ILearner {

	public void learn(DataSet trainingSet);
	
	public void resetToUnlearnedState();
	
	public NeuralNetwork getNeuralNetwork();
}
