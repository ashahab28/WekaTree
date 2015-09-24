/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package WekaInterface;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.misc.SerializedClassifier;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.filters.Filter;
/**
 * Source code refer to WekaDemo.java by FracPete(fracpete at waikato dot ac dot nz)
 * Documentation of weka: weka.wikispaces.com, weka.sourceforge.net
 * @author ahmadshahab
 */
public class Weka {
    
    private Classifier m_Classifier;
    
    private Filter m_Filter;
    
    private String m_TrainingFile;
    
    private Instances m_Training;
    
    private Evaluation m_Evaluation;
    
    public Weka(){
        /* Constructor */
    }
    
    /**
     * Build Weka classifier
     * @param name The type of the classifier
     * @param options The set of options will be given to the classifier
     * @throws Exception 
     */
    public void setClassifier(String name, String[] options) throws Exception{
        m_Classifier = AbstractClassifier.forName(name, options);
    }
    
    /**
     * Build Weka filter
     * @param name Class name of the filter
     * @param options The set of options will be given to the filter
     * @throws Exception 
     */
    public void setFilter(String name, String[] options) throws Exception{
        m_Filter = (Filter) Class.forName(name).newInstance();
        if(m_Filter instanceof OptionHandler)
            ((OptionHandler) m_Filter).setOptions(options);
    }
    
    /**
     * Load dataset for training
     * @param filename The file for training set : arff or csv format
     * @throws IOException 
     */
    public void setTraining(String filename) throws IOException{
        m_TrainingFile = filename;
        m_Training = new Instances(new BufferedReader(new FileReader(m_TrainingFile)));
        // By default set the class index to the last attribute
        m_Training.setClassIndex(m_Training.numAttributes() - 1);
    }
    
    /**
     * Remove the specified attribute from the current Instance
     * @param indexes numbers of attribute to be removed
     */
    public void removeAttribute(int indexes[]){
        if(indexes.length < m_Training.numAttributes()){
            System.out.println("Error! Number of index > num of attribute.");
        }
        else{
            for(int i = 0; i < indexes.length; i++){
                m_Training.remove(indexes[i]);
            }
        }
    }
    
    /**
     * Run the classifier using given filter parameter
     * and training instances to build the training model
     * @param summary print the summary result if the value is true
     * @throws java.lang.Exception
     */
    public void run(boolean summary) throws Exception{
        // Run filter, NEXT TO-DO is handle the exception if there is no filter
        m_Filter.setInputFormat(m_Training);
        Instances filtered = Filter.useFilter(m_Training, m_Filter);
        
        // Train the classifier
        m_Classifier.buildClassifier(filtered);
        
        // Evaluation, use 10 Cross Validation
        m_Evaluation = new Evaluation(filtered);
        m_Evaluation.crossValidateModel(m_Classifier, filtered, 10, m_Training.getRandomNumberGenerator(1));
        
        if(summary){
            System.out.println(m_Evaluation.toSummaryString("10 Cross Validation Result", true));
            System.out.println(m_Evaluation.toMatrixString());
        }
    }
    
    /**
     * Save the training model
     * @param filename name of the saved model 
     */
    public void saveModel(String filename){
        Debug.saveToFile(filename, m_Classifier);
    }
    
    /**
     * Load the model to filename
     * @param filename 
     */
    public void loadModel(String filename){
        SerializedClassifier classifier = new SerializedClassifier();
        classifier.setModelFile(new File(filename));
        
        m_Classifier = classifier.getCurrentModel();
    }
    
    /**
     * Test the trained model using the test-set file
     * @param filename 
     * @throws java.lang.Exception 
     */
    public void testModel(String filename) throws Exception{
        Instances test = new Instances(new BufferedReader(new FileReader(filename)));
        test.setClassIndex(test.numAttributes() - 1);
        m_Evaluation.evaluateModel(m_Classifier, test);
        
        System.out.println(m_Evaluation.toSummaryString("Testing Result", true));
        System.out.println(m_Evaluation.toMatrixString());
    }
    
    /**
     * Classify unlabeled data and save the result
     * @param input unlabeled data
     * @param output classified data
     * @throws java.lang.Exception
     */
    public void classifyInstance(String input, String output) throws Exception{
        Instances unlabeled = new Instances(new BufferedReader(new FileReader(input)));
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        
        // Create a copy
        Instances labeled = new Instances(unlabeled);
        
        for(int i = 0; i < unlabeled.numInstances(); i++){
            double clsLabel = m_Classifier.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        
        try ( 
            // Save the result
            BufferedWriter writer = new BufferedWriter(new FileWriter(output))) {
            writer.write(labeled.toString());
            writer.newLine();
            writer.flush();
        }
    }
    
    public static void main(String[] args) throws Exception{
        Weka weka = new Weka();
        String[] options_cl = {"-U"};
        weka.setClassifier("weka.classifiers.trees.J48", options_cl);
        
        String[] options_f= {""};
        weka.setFilter("weka.filters.unsupervised.instance.Randomize", options_f);
        
        weka.setTraining("weather.nominal.arff");
        weka.run(false);
    }
}   
