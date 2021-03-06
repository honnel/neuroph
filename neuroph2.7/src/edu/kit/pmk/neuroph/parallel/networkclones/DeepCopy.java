package edu.kit.pmk.neuroph.parallel.networkclones;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;

public class DeepCopy {

	private DeepCopy() {
	}

	/**
	 * Quote: http://javatechniques.com/blog/faster-deep-copies-of-java-objects/ Picture 6
	 * 
	 * Utility for making deep copies (vs. clone()'s shallow copies) of
	 * objects. Objects are first serialized and then deserialized. Error
	 * checking is fairly minimal in this implementation. If an object is
	 * encountered that cannot be serialized (or that references an object
	 * that cannot be serialized) an error is printed to System.err and
	 * null is returned. Depending on your specific application, it might
	 * make more sense to have copy(...) re-throw the exception.
	 */
	public static Object createDeepCopy(Object original) throws IOException,
			ClassNotFoundException {
		long t0 = System.currentTimeMillis();
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

		long t1 = System.currentTimeMillis();
		System.out.println("Deep Copy took " + (t1-t0) + " ms.");
		return copy;
	}
}