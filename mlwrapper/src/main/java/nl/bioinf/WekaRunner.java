package nl.bioinf;


import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;

public class WekaRunner {

    public static void main(String[] args) throws Exception {
        WekaRunner wekaRunner = new WekaRunner();
        wekaRunner.start(args);
    }

    private void start(String[] args) throws Exception {
        AbstractClassifier fromFile = loadClassifier();
        Instances unknownInstances = loadarff(args[0]);
        classifyNewInstance(fromFile, unknownInstances);
    }

    private void classifyNewInstance(AbstractClassifier model, Instances unknownInstances) throws Exception {
        Instances labeled = new Instances(unknownInstances);
        for (int i = 0; i < unknownInstances.numInstances(); i++) {
            double clsLabel = model.classifyInstance(unknownInstances.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        String ouput = labeled.toString();
        ouput = ouput.split("@data")[1];
        ouput = ouput.strip();
        String[] lines = ouput.split(System.getProperty("line.separator"));
        int counter = 1;
        for (String line:lines) {
            if (line.endsWith("1")) {
                System.out.println("Sample" + counter + ": Cancer Detected");
            } else {
                System.out.println("Sample" + counter + ": Healthy");

            }
            counter++;
        }
    }

    private Instances loadarff(String datafile) throws IOException {
        try {
            DataSource source = new DataSource(datafile);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            return data;

        } catch (Exception e) {
            throw new IOException("Reading file gave an error!");
        }
    }


    private AbstractClassifier loadClassifier() throws Exception {
        String modelFile = "model/model.model";
        return (AbstractClassifier) weka.core.SerializationHelper.read(modelFile);
    }
}
