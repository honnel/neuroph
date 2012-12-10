package edu.kit.pmk.neuroph.eval.generality;

public class Score {
	
	public final double error;
	public final long time;
	
	public Score(double error, long time) {
		this.error = error;
		this.time = time;
	}

}
