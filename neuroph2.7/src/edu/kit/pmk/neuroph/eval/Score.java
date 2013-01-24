package edu.kit.pmk.neuroph.eval;

import edu.kit.pmk.neuroph.parallel.ILearner;

public class Score {

	public double[] errors;
	public long[] times;
	public final ILearner learner;

	public Score(int runs, ILearner learner) {
		this.errors = new double[runs];
		this.times = new long[runs];
		this.learner = learner;
	}

	public long getAverageTime() {
		return getOverallTime() / times.length;
	}

	public double getAverageError() {
		return getSummedUpError() / times.length;
	}

	public long getOverallTime() {
		long sum = 0;
		for (long t : times) {
			sum += t;
		}
		return sum;
	}

	public double getSummedUpError() {
		double sum = 0;
		for (double e : errors) {
			sum += e;
		}
		return sum;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("SCORE [" + learner.getDescription() + "]");
		sb.append(String.format("average error=%f, average time=%dms",
				getAverageError(), getAverageTime()));
		sb.append(String.format("summed up error=%f, overall time=%dms",
				getSummedUpError(), getOverallTime()));
		for (int i = 0; i < times.length; i++) {
			sb.append(String.format("Run-%i: error=%f, time=%dms", errors[i],
					times[i]));
		}
		sb.append("\n");
		return sb.toString();
	}
}
