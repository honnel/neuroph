package edu.kit.pmk.neuroph.eval.generality;

import org.neuroph.core.learning.DataSet;

public class TestAndTrainingSet {

	private DataSet trainingSet;
	private DataSet testSet;

	private TestAndTrainingSet(int inputSize, int outputSize) {
		this.trainingSet = new DataSet(inputSize, outputSize);
		this.testSet = new DataSet(inputSize, outputSize);
	}

	public static TestAndTrainingSet splitSet(DataSet dataSet,
			double trainingSetRatio) {
		int splitter = (int) (trainingSetRatio * dataSet.size());
		TestAndTrainingSet tats = new TestAndTrainingSet(
				dataSet.getInputSize(), dataSet.getOutputSize());
		tats.trainingSet.getRows().addAll(
				dataSet.getRows().subList(0, splitter));
		tats.testSet.getRows().addAll(
				dataSet.getRows().subList(splitter, dataSet.size()));
		return tats;
	}

	public static TestAndTrainingSet splitSetAndPermute(DataSet dataSet,
			double trainingSetRatio) {
		DataSet permutedSet = new DataSet(dataSet.getInputSize(),
				dataSet.getOutputSize());
		permutedSet.getRows().addAll(dataSet.getRows());
		permutedSet.shuffle();
		return splitSet(permutedSet, trainingSetRatio);
	}

	public DataSet getTrainingSet() {
		return trainingSet;
	}

	public DataSet getTestSet() {
		return testSet;
	}
}
