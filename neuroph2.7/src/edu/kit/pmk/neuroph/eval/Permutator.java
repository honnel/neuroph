package edu.kit.pmk.neuroph.eval;

import java.util.Random;

public class Permutator {

	private Permutator() {
	}

	public static int[] getPermutation(int size) {
		Random r = new Random();
		int[] perm = new int[size];
		boolean[] taken = new boolean[size];
		for (int i = 0; i < size; i++) {
			int num;
			do {
				num = r.nextInt(size);
			} while (taken[num]);
			taken[num] = true;
			perm[i] = num;
		}
		return perm;
	}
}