package edu.kit.pmk.neuroph.parallel.IrisClassificationSample;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class DeepCopy {

	private DeepCopy() {
	}

	public static Object createDeepCopy(Object original) throws IOException,
			ClassNotFoundException {
		Object copy;

		// Write the object out to a byte array
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream out = new ObjectOutputStream(bos);
		out.writeObject(original);
		out.close();

		// Make an input stream from the byte array and read
		// a copy of the object back in.
		ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(
				bos.toByteArray()));
		copy = in.readObject();
		in.close();

		return copy;
	}
}