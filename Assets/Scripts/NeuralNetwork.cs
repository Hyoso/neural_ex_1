using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

[RequireComponent(typeof(Topology))]
public class NeuralNetwork : MonoBehaviour
{
    Topology m_networkTopology;
    List<NeuralLayer> m_layers = new List<NeuralLayer>();

    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;

	void Start ()
    {
        m_networkTopology = GetComponent<Topology>();

        // construct layers
        List<double> feedForwardInput = new List<double>();
        int layerCount = m_networkTopology.m_layers.Count;
        // for each index in network topolgy, create new layer
        for (int i = 0; i < layerCount; i++)
        {
            m_layers.Add(new NeuralLayer());
            // if this is the last layer then there are no connections, so return 0. Otherwise find the number of neurons in the next layer
            uint neuronConnections = i == m_networkTopology.m_layers.Count - 1 ? 0 : m_networkTopology.m_layers[i + 1];

            // add neurons to layer
            // using "<=" so we have an extra layer for the bias values
            for (int j = 0; j <= m_networkTopology.m_layers[i]; j++)
            {
                m_layers[i].m_neurons.Add(new Neuron(neuronConnections, j));
            }

            // set bias value of this layer
            m_layers[i].m_neurons[m_layers[i].m_neurons.Count - 1].SetOutputValue(1.0);
        }

        List<double> inputVals = new List<double>(); // for feedforwarding
        List<double> targetVals = new List<double>(); // for backprop
        List<double> results = new List<double>();

        inputVals.Add(1.0);
        inputVals.Add(0.0);
        FeedForward(inputVals);
        return;
        BackPropogation(targetVals);
        GetResults(results);
    }

    void FeedForward(List<double> input)
    {
        if (input == null || input.Count == 0)
        {
            Debug.LogError("invalid feedforward list");
            return;
        }

        // set the first layers neuron inputs
        for (int i = 0; i < input.Count; i++)
        {
            m_layers[0].m_neurons[i].SetOutputValue(input[i]);
        }

        // for each layer (starting from the 2nd layer)
        for (int i = 1; i < m_layers.Count; i++)
        {
            // for each neuron
            for (int j = 0; j < m_layers[i].m_neurons.Count - 1; j++) // minus one from count because we ignore the bias neuron
            {
                m_layers[i].m_neurons[j].FeedForward(m_layers[i - 1]);
            }
        }
    }

    /// <summary>
    /// calculate overall net error (RMS(Root Means Square) errors formula)
    /// </summary>
    /// <param name="targetValues"></param>
    void BackPropogation(List<double> targetValues)
    {
        NeuralLayer outputLayer = m_layers[m_layers.Count - 1];
        m_error = 0.0;

        // calculate rms error for the last (output) layer
            // rms = average of (sum of (target - actualvalue squared))
            // rms = sqrt(1 / (target - output value)squared)
        for (int i = 0; i < outputLayer.m_neurons.Count; i++)
        {

            double errorDelta = targetValues[i] - outputLayer.m_neurons[i].GetOutputValue();
            m_error += errorDelta * errorDelta;
        }
        m_error /= outputLayer.m_neurons.Count - 1;
        m_error = Math.Sqrt(m_error);

        // gives output of the current error avg
        // allows us to check how close/far the network is to completion
        m_recentAverageError = ((m_recentAverageError * m_recentAverageSmoothingFactor) + m_error) / (m_recentAverageSmoothingFactor + 1.0);

        // calculate output layer gradients
        for (int i = 0; i < outputLayer.m_neurons.Count; i++)
        {
            outputLayer.m_neurons[i].CalculateOutputGradients(targetValues[i]);
        }

        // calculate gradients on hidden (a layer between first and last layers) layers
        // starting from outputlayerIndex - 1
        // and moving back to the initiallayerIndex + 1
        for (int i = m_layers.Count - 2; i > 0; i--)
        {
            NeuralLayer curHiddenLayer = m_layers[i];
            NeuralLayer nextLayer = m_layers[i + 1];

            for (int j = 0; j != curHiddenLayer.m_neurons.Count; j++)
            {
                curHiddenLayer.m_neurons[j].CalculateHiddenGradients(nextLayer);
            }
        }

        // for all layers from outputs to first hidden layer
            // update connection weights
        for (int i = m_layers.Count - 1; i > 0; i--)
        {
            NeuralLayer curLayer = m_layers[i];
            NeuralLayer prevLayer = m_layers[i - 1];

            for (int j = 0; j != curLayer.m_neurons.Count; j++)
            {
                curLayer.m_neurons[j].UpdateInputWeights(prevLayer);
            }
        }
    }

    void GetResults(List<double> results)
    {
        results.Clear();

        int outputLayerNeuronCount = m_layers[m_layers.Count - 1].m_neurons.Count;
        for (int i = 0; i < outputLayerNeuronCount; i++)
        {
            results.Add(m_layers[m_layers.Count - 1].m_neurons[i].GetOutputValue());
        }

        for (int i = 0; i != results.Count; i++)
        {
            print(results[i]);
        }
    }
}
