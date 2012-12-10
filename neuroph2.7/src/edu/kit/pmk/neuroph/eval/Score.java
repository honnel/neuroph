package edu.kit.pmk.neuroph.eval;

public class Score {

	public final double error;
	public final long time;

	public Score(double error, long time) {
		this.error = error;
		this.time = time;
	}

	@Override
	public String toString() {
		return String.format("Score(error=%f, time=%dms)", error, time);
	}

}
