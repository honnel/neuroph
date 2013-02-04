package edu.kit.pmk.neuroph.eval;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

public class CommandLineInterface {

	private static CommandLineParser parser;
	private static Options options;

	static {
		options = new Options();
		parser = new PosixParser();
		Option cf = new Option("cf", "config", true,
				"Neuroph experiment configuration file");
		cf.setRequired(true);
		options.addOption(cf);
		Option o = new Option("o", "outputdir", true, "Output file directory");
		o.setRequired(true);
		options.addOption(o);
		options.addOption("id", true,
				"Test identifier");
		
		options.addOption("v", "verbose", false,
				"Print everything to command line");
		options.addOption("d", "debug", false, "Write info and debug output");
		options.addOption("i", "info", false, "Write only info output");
		options.addOption("csv", false, "Write csv output");
	}

	public static CommandLine getCommandLine(String[] args) {
		try {
			return parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
			throw new IllegalArgumentException("Argument parsing failed!");
		}
	}

}
