package org.neuroph.nnet.learning;

import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.learning.DataSet;

public class BatchParallelSlave extends MomentumBackpropagation {

	private boolean onStartCalled = false;



	@Override
	public void setTrainingSet(DataSet trainingSet) {
		super.setTrainingSet(trainingSet);
		super.onStart();
	}

	@Override
	public void learn(DataSet trainingSet) {

		beforeEpoch();
		doLearningEpoch(trainingSet);
		this.currentIteration++;
	}
}
