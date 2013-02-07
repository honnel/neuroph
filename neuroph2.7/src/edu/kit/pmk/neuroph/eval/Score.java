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

	/**
	 * Calculates the alpha 95% confidence interval
	 * 
	 * @return array with two elements, first lo border, second hi border
	 */
	public long[] getConfidenceIntervalTime() {
		long[] result = new long[2];
		result[0] = 0;
		result[1] = 0;
		if (times.length > 1) {
			long xxbar = 0;
			long avgTime = getAverageTime();
			for (int i = 0; i < times.length; i++) {
				xxbar += (times[i] - avgTime) * (times[i] - avgTime);
			}
			long variance = xxbar / (times.length - 1);
			long stddev = (long) Math.sqrt(variance);
			long lo = (long) (avgTime - 1.96 * stddev);
			long hi = (long) (avgTime + 1.96 * stddev);
			result[0] = lo;
			result[1] = hi;
		}
		return result;
	}

	public double[] getConfidenceIntervalError() {
		double[] result = new double[2];
		result[0] = 0.0;
		result[1] = 0.0;
		if (errors.length > 1) {
			double xxbar = 0.0;
			double avgError = getAverageError();
			for (int i = 0; i < errors.length; i++) {
				xxbar += (errors[i] - avgError) * (errors[i] - avgError);
			}
			double variance = xxbar / (errors.length - 1);
			double stddev = Math.sqrt(variance);
			double lo = (avgError - 1.96 * stddev);
			double hi = (avgError + 1.96 * stddev);
			result[0] = lo;
			result[1] = hi;
		}
		return result;
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
			sum += Math.abs(e);
		}
		return sum;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		long[] confidenceIntervallTime = getConfidenceIntervalTime();
		double[] confidenceIntervallError = getConfidenceIntervalError();
		sb.append("SCORE [" + learner.getDescription() + "]");
		sb.append(" ThreadCount=" + learner.getNumberOfThreads());
		sb.append(String.format(
				" average error=%f [%f;%f], average time=%dms [%d;%d]",
				getAverageError(), confidenceIntervallError[0],
				confidenceIntervallError[1], getAverageTime(),
				confidenceIntervallTime[0], confidenceIntervallTime[1]));
		sb.append(String.format(", summed up error=%f, overall time=%dms",
				getSummedUpError(), getOverallTime()));
		for (int i = 0; i < times.length; i++) {
			sb.append(String.format(", Run-%d: error=%f, time=%dms", i,
					errors[i], times[i]));
		}
		return sb.toString();
	}

	public String toCsv() {
		StringBuilder sb = new StringBuilder();
		long[] confidenceIntervallTime = getConfidenceIntervalTime();
		double[] confidenceIntervallError = getConfidenceIntervalError();
		sb.append(String.format("%s;%d;%d;%f;%f;%f;%d;%d;%d;", learner.getClass().getSimpleName(), learner.getNumberOfThreads(),errors.length,getAverageError(),
				confidenceIntervallError[0], confidenceIntervallError[1],
				getAverageTime(), confidenceIntervallTime[0],
				confidenceIntervallTime[1]));
		sb.append(String.format("%f;%d", getSummedUpError(), getOverallTime()));
		return sb.toString();
	}
}
