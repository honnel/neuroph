package edu.kit.pmk.neuroph.samples.CernParticleCollision;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

public class CernFormatConverter {

	private static final char REPLACEMENT = '#';
	private String keysInputsFilepath;
	private String passingKeysFilepath;
	private HashMap<Integer, DataRow> keyToInput;

	public CernFormatConverter(String keysInputsFilepath,
			String passingKeysFilepath) {
		this.keysInputsFilepath = keysInputsFilepath;
		this.passingKeysFilepath = passingKeysFilepath;
	}

	public static void main(String[] args) throws Exception {
		// System.out.println("".split(",").length);
		// System.out.println("2,3,".split(",").length);
		// String line = replaceCommasInParentheses("(2,3,4),8.0,(-34.2,)");
		// System.out.println(line);
		// for (String value : line.split(",")) {
		// System.out.println(countElements(value));
		// }

		CernFormatConverter cfc = new CernFormatConverter(
				"data/cern/result.txt", "data/cern/eventsPassSelectionExample");
		cfc.writeToFile("data/cern/converted.txt");
		System.out.println(cfc.getInputCount());
	}

	public int getInputCount() throws IOException {
		if (keyToInput == null) {
			prepareData();
		}
		return keyToInput.values().iterator().next().values.length;
	}

	public int getOutputCount() throws IOException {
		// :p
		return 1;
	}

	public void writeToFile(String outputFilepath) throws IOException {
		BufferedWriter fWriter = new BufferedWriter(new FileWriter(
				outputFilepath));
		if (keyToInput == null) {
			prepareData();
		}

		for (Entry<Integer, DataRow> e : keyToInput.entrySet()) {
			DataRow dr = e.getValue();
			for (double d : dr.values) {
				fWriter.write(d + ",");
			}
			fWriter.write(dr.output + System.lineSeparator());
		}

		fWriter.close();
	}

	private void prepareData() throws IOException {
		int[] elementsInColumn = countElementsPerColumn(keysInputsFilepath);
		keyToInput = parseInputs(keysInputsFilepath, elementsInColumn);
		setOutputs();
	}

	private void setOutputs() throws IOException {
		BufferedReader fReader = new BufferedReader(new FileReader(
				passingKeysFilepath));
		String line;
		while ((line = fReader.readLine()) != null) {
			Integer key = Integer.parseInt(line);
			DataRow dr = keyToInput.get(key);
			if (dr != null)
				dr.output = 1;
		}
		fReader.close();
	}

	private HashMap<Integer, DataRow> parseInputs(String keysInputsFilepath,
			int[] elementsInColumn) throws IOException {
		int rowCount = elementsInColumn[0];
		HashMap<Integer, DataRow> inputs = new HashMap<>(rowCount);
		BufferedReader fReader = new BufferedReader(new FileReader(
				keysInputsFilepath));

		String line;

		while ((line = fReader.readLine()) != null) {
			StringBuilder lineInputs = new StringBuilder();
			String rLine = replaceCommasInParentheses(line);
			String[] values = rLine.split(",");
			Integer key = Integer.parseInt(values[0]);
			for (int i = 1; i < elementsInColumn.length; i++) {
				lineInputs.append(padWithZeros(values[i], elementsInColumn[i]));
			}
			inputs.put(key, new DataRow(lineInputs.toString()));
		}

		fReader.close();
		return inputs;
	}

	private String padWithZeros(String value, int targetCount) {
		String s = value.replaceAll("\\(", "");
		s = s.replaceAll("\\)", "");
		s = s.replaceAll("" + REPLACEMENT, ",");
		StringBuilder inputs = new StringBuilder(s);
		if (!s.equals("") && !s.endsWith(",")) {
			inputs.append(',');
		}
		int count = countElements(inputs.toString());
		assert count <= targetCount : "Maximum column count is wrong.";
		for (int i = 0; i < (targetCount - count); i++) {
			inputs.append("0,");
		}
		return inputs.toString();
	}

	// elementsPerColumn[0] = #lines in File
	private int[] countElementsPerColumn(String keysInputsFilepath)
			throws IOException {
		BufferedReader fReader = new BufferedReader(new FileReader(
				keysInputsFilepath));

		String line = fReader.readLine();
		String rLine = replaceCommasInParentheses(line);
		String[] values = rLine.split(",");
		int[] elementsPerColumn = new int[values.length];

		do {
			elementsPerColumn[0]++;
			rLine = replaceCommasInParentheses(line);
			values = rLine.split(",");
			for (int i = 1; i < elementsPerColumn.length; i++) {
				elementsPerColumn[i] = Math.max(elementsPerColumn[i],
						countElements(values[i]));
			}
		} while ((line = fReader.readLine()) != null);

		fReader.close();
		return elementsPerColumn;
	}

	String replaceCommasInParentheses(String line) {
		StringBuffer sb = new StringBuffer(line);
		boolean betweenParentheses = false;
		for (int i = 0; i < line.length(); i++) {
			if (line.charAt(i) == '(')
				betweenParentheses = true;
			else if (line.charAt(i) == ')')
				betweenParentheses = false;
			else if (betweenParentheses && (line.charAt(i) == ',')) {
				sb.setCharAt(i, REPLACEMENT);
			}
		}
		return sb.toString();
	}

	private int countElements(String value) {
		if (value == null || value.equals(""))
			return 0;
		else if (value.matches("\\(.*\\)"))
			return value.substring(1, value.length() - 1).split(
					"" + REPLACEMENT).length;
		else
			return value.split(",").length;
	}

	class DataRow {
		double[] values;
		double output = 0;

		public DataRow(int length) {
			this.values = new double[length];
		}

		public DataRow(String string) {
			String[] doubles = string.split(",");
			this.values = new double[doubles.length];
			for (int i = 0; i < values.length; i++) {
				values[i] = Double.parseDouble(doubles[i]);
			}
		}
	}
}
