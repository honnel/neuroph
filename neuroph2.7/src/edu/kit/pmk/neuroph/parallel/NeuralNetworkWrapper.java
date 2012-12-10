package edu.kit.pmk.neuroph.parallel;

import java.io.IOException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;

import edu.kit.pmk.neuroph.parallel.networkclones.FastDeepCopy;

public class NeuralNetworkWrapper implements ILearner {
	
	private NeuralNetwork originalNet;
	private NeuralNetwork neuralNet;
	private int numThreads;
	
	public NeuralNetworkWrapper(NeuralNetwork neuralNet, int numThreads) {
		this.originalNet = neuralNet;
		this.numThreads = numThreads;
		resetToUnlearnedState();
	}

	
	@Override
	public void learn(DataSet trainingSet) {
		neuralNet.learn(trainingSet);
	}

	@Override
	public void resetToUnlearnedState() {
		try {
			this.neuralNet =  (NeuralNetwork) FastDeepCopy.createDeepCopy(originalNet);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}

	@Override
	public NeuralNetwork getNeuralNetwork() {
		return neuralNet;
	}

	@Override
	public int getNumberOfThreads() {
		return numThreads;
	}

}
