package edu.kit.pmk.neuroph.eval;

import edu.kit.pmk.neuroph.parallel.ILearner;

public class Score {

	public double error;
	public long time;
	public final ILearner learner;

	public Score(double error, long time, ILearner learner) {
		this.error = error;
		this.time = time;
		this.learner = learner;
	}

	@Override
	public String toString() {
		return String.format("%s Score(error=%f, time=%dms)", learner.getDescription(), error, time);
	}
}
