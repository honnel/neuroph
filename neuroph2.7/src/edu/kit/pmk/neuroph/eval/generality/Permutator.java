package edu.kit.pmk.neuroph.eval.generality;

import java.util.Random;

public class Permutator {

	private Permutator() {
	}

	public static int[] getPermutation(int n) {
		Random r = new Random();
		int[] perm = new int[n];
		boolean[] taken = new boolean[n];
		for (int i = 0; i < n; i++) {
			int num;
			do {
				num = r.nextInt(n);
			} while (taken[num]);
			taken[num] = true;
			perm[i] = num;
		}
		return perm;
	}
}