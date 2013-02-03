package edu.kit.pmk.neuroph.eval.experiment;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

public class ExperimentConfiguration {

	private HashMap<String, String> args;
	private String name;
	private final String DELIMITER = ":";

	public ExperimentConfiguration(String name) {
		this.name = name;
		this.args = new HashMap<String, String>();
	}

	public ExperimentConfiguration(String name, String filepath) {
		this.name = name;
		this.args = new HashMap<String, String>();
		try {
			parseTestConfigurationFile(filepath);
		} catch (IOException e) {
			System.err.println("Test Configuration file parsing failed.");
			e.printStackTrace();
		}
	}

	private void parseTestConfigurationFile(String filepath) throws IOException {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(filepath));
			String line = "";
			while ((line = reader.readLine()) != null) {
				String[] keyValue = line.split(DELIMITER);
				setArgument(keyValue[0].trim(), keyValue[1].trim());
			}
		} finally {
			reader.close();
		}
	}

	public void setArgument(ExperimentConfigurationArgument arg, String value) {
		args.put(arg.name(), value);
	}

	public void setArgument(String arg, String value) {
		args.put(arg, value);
	}

	public String getArgument(ExperimentConfigurationArgument arg) {
		return args.get(arg.name());
	}

	public String getArgument(String arg) {
		return args.get(arg);
	}

	public String getName() {
		return name;
	}

	private final static String EOL = System.lineSeparator();

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("+-+ TestConfiguration '" + this.name + "' +-+" + EOL);
		for (Entry<String, String> e : args.entrySet()) {
			sb.append(e.getKey() + ": " + e.getValue() + EOL);
		}
		sb.append(EOL);
		return sb.toString();
	}

}
