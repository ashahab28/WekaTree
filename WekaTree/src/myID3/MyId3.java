/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myID3;

import WekaInterface.Weka;
import java.io.IOException;
import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 *
 * @author ahmadshahab
 */
public class MyId3 extends AbstractClassifier{
    
    /** Attribute of this class */
    
    private MyId3[] nodes;
    
    // For splitting the tree
    private Attribute currentAttribute;
    
    // If leaf, then the value is class value
    private double classValue;
    
    // Class distribution (if leaf)
    private double[] classDistribution;
    
    // Attribute identity for the class of the node (if leaf)
    private Attribute classAttribute;
    
    /**
     * Build an Id3 classifier
     * @param instances dataset used for building the model
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
        // Detecting the instance type, can Id3 handle the data?
        getCapabilities().testWithFail(instances);
        
        // Remove missing class
        Instances data = new Instances(instances);
        data.deleteWithMissingClass();
        
        // Build the id3
        buildTree(data);
    }
    
    /**
     * Construct the tree using the given instance
     * Find the highest attribute value which best at dividing the data
     * @param data Instance
     */
    public void buildTree(Instances data){
        if(data.numInstances() > 0){
            // Lets find the highest Information Gain!
            // First compute each information gain attribute
            double IG[] = new double[data.numAttributes()];
            Enumeration enumAttribute = data.enumerateAttributes();
            while(enumAttribute.hasMoreElements()){
                Attribute attribute = (Attribute)enumAttribute.nextElement();
                IG[attribute.index()] = informationGain(data, attribute);
            }
            // Assign it as the tree attribute!
            currentAttribute = data.attribute(Utils.maxIndex(IG));
            
            // IG = 0 then current node = leaf!
            if(Utils.eq(IG[currentAttribute.index()], 0)){
                // Set the class value as the highest frequency of the class
                currentAttribute = null;
                classDistribution = new double[data.numClasses()];
                Enumeration enumInstance = data.enumerateInstances();
                while(enumInstance.hasMoreElements()){
                    Instance temp = (Instance)enumInstance.nextElement();
                    classDistribution[(int) temp.classValue()]++;
                }
                Utils.normalize(classDistribution);
                classValue = Utils.maxIndex(classDistribution);
                classAttribute = data.classAttribute();
            }
            else{
                // Create another node from the current tree
                Instances[] splitData = splitDataByAttribute(data, currentAttribute);
                nodes = new MyId3[currentAttribute.numValues()];
                for (int i = 0; i < currentAttribute.numValues(); i++) {
                    nodes[i] = new MyId3();
                    System.out.println();
                    nodes[i].buildTree(splitData[i]);
                }
            }
        }
        else{
            classAttribute = null;
            classValue = Utils.missingValue();
            classDistribution = new double[data.numClasses()];
        }
    }
    
    /**
     * Count the information gain for selected attribute 
     * from the given dataset
     * @param data
     * @param attribute
     * @return 
     */
    private double informationGain(Instances data, Attribute attribute){
        
        /* Information Gain = Init entropy - After change entropy */
        
        double initEntropy = entropy(data);
        
        // Now we split the attribute first to count each entropy on different value
        Instances[] subSet = splitDataByAttribute(data, attribute);
        double[] entropy = new double[attribute.numValues()];
        
        // Count the entropy!
        for(int i = 0; i < attribute.numValues(); i++){
            if(subSet[i].numInstances() > 0) entropy[i] = entropy(subSet[i]);
            else entropy[i] = 0;
        }
        
        double IG = initEntropy;
        for (int i = 0; i < attribute.numValues(); i++) {
            IG = IG - (entropy[i]*(double)subSet[i].numInstances()/data.numInstances());
        }
        
        return IG;
    } 
    
    /**
     * Find the entropy from a given dataset
     * @param data
     * @return 
     */
    private double entropy(Instances data){
    
        /*  Entropy = -(p1 log2 p1) -(p2 log2 p2).... */
        
        double numInstance = data.numInstances();
        double numClass = data.numClasses();
        double[] distribution = new double[data.numClasses()];
        
        Enumeration instance = data.enumerateInstances();
        while(instance.hasMoreElements()){
            Instance temp = (Instance) instance.nextElement();
            /* Count the p1, p2 */
            distribution[(int)temp.classValue()] ++;
        }

        /* Sum all the distribution */
        double sum = 0;
        for(int i = 0; i < numClass; i++){
            distribution[i] = distribution[i]/numInstance;
            distribution[i] *= Utils.log2(distribution[i]);
            sum += distribution[i];
        }

        return -1 * sum;
    }
    
    /**
     * Create split of data based on the value of attribute
     * @param data
     * @param attribute
     * @return 
     */
    private Instances[] splitDataByAttribute(Instances data, Attribute attribute){
        
        // Init the object first
        Instances[] subSet = new Instances[attribute.numValues()];
        for (int i = 0; i < attribute.numValues(); i++) {
            subSet[i] = new Instances(data, data.numInstances());
        }
        
        // Split it!
        Enumeration instanceEnum = data.enumerateInstances();
        while(instanceEnum.hasMoreElements()){
            Instance instance = (Instance)instanceEnum.nextElement();
            subSet[(int)instance.value(attribute)].add(instance);    
        }
        
        // Compact the array of object by removing the empty array
        for (int i = 0; i < attribute.numValues(); i++) {
            subSet[i].compactify();
        }
        
        return subSet;
    }
    
    /**
     * Capability of id3 classifier
     * @return 
     */
    @Override
    public Capabilities getCapabilities(){
        Capabilities id3_capability = super.getCapabilities();
        id3_capability.disableAll();
        
        // Attribute type capability
        id3_capability.enable(Capability.NOMINAL_ATTRIBUTES);
        
        // Class capability
        id3_capability.enable(Capability.NOMINAL_CLASS);
        id3_capability.enable(Capability.MISSING_CLASS_VALUES);
        
        // Minimum number of instances allowed to be use
        id3_capability.setMinimumNumberInstances(0);
        
        return id3_capability;
    }
    
    /**
   * Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                   + "please.");
    }
    if (currentAttribute == null) {
      return classValue;
    } else {
      return nodes[(int) instance.value(currentAttribute)].
        classifyInstance(instance);
    }
  }

  /**
   * Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                   + "please.");
    }
    if (currentAttribute == null) {
      return classDistribution;
    } else { 
      return nodes[(int) instance.value(currentAttribute)].
        distributionForInstance(instance);
    }
  }
  
  public static void main(String[] args) throws IOException, Exception{
      Weka a = new Weka();
      a.setTraining("weather.nominal.arff");
      Classifier b = new MyId3();
      b.buildClassifier(a.getM_Training());
  }
  
//                outlook = sunny
//              |  humidity = high: no
//              |  humidity = normal: yes
//              outlook = overcast: yes
//              outlook = rainy
//              |  windy = TRUE: no
//              |  windy = FALSE: yes
}
