using System.Collections.Generic;
using System;

public class Neuron
{
    public List<double> m_deltaWeights; // difference in weights when doing forward/backprop
    public List<double> m_outputWeights;
    private double m_output;
    private int m_myIndex;
    private double m_gradient;

    // learning rate, 0.0 = slow, 0.2 = medium, 1.0 = fast
    private const double CONST_ETA = 0.15;

    // momentum, how much i improve from last delta weight
    private const double CONST_ALPHA = 0.1;

    /// <summary>
    /// Creates a new neuron with the specified outputs
    /// </summary>
    /// <param name="outputCount"> lets this neuron know how many other neurons it needs to output its data to </param>
    public Neuron(uint outputCount, int myIndex)
    {
        for (int i = 0; i != outputCount; i++)
        {
            m_outputWeights.Add(RandomWeight());
            m_deltaWeights.Add(RandomWeight());
        }

        m_myIndex = myIndex;
    }
    /// <summary>
    /// creates a random double value
    /// </summary>
    /// <returns>a double between 0 and 1 inclusive</returns>
    public double RandomWeight()
    {
        return new Random().NextDouble() / double.MaxValue;
    }

    public void SetOutputValue(double val)
    {
        m_output = val;
    }

    public double GetOutputValue()
    {
        return m_output;
    }


    /// <summary>
    /// main calculation functions
    /// takes in output from previous layer
    /// sums it
    /// outputs a new modified value
    /// </summary>
    /// <param name="previousLayer"></param>
    public void FeedForward(NeuralLayer previousLayer)
    {
        double sum = 0.0;
        for (int i = 0; i != previousLayer.m_neurons.Count; i++)
        {
            // multiple the previous layers output value (this current neurons input) by previous layers outputweight to me
            // and add to the sum
            sum += previousLayer.m_neurons[i].GetOutputValue() * previousLayer.m_neurons[i].m_outputWeights[m_myIndex];
        }

        m_output = Transfer(sum);
    }


    /// <summary>
    /// for forward propogation
    /// </summary>
    /// <param name="x"></param>
    /// <returns>value between [-1...1]</returns>
    public double Transfer(double x)
    {
        // using tanh but can use ramp? exp? sigmoid? step?
        return Math.Tanh(x);
    }

    /// <summary>
    /// for back propogation
    /// </summary>
    /// <param name="x"></param>
    /// <returns>derivative of tanh</returns>
    public double TransferDerivative(double x)
    {
        return 1.0 - (x * x);
    }

    public void CalculateOutputGradients(double targetValue)
    {
        double delta = targetValue - m_output;
        m_gradient = delta * TransferDerivative(m_output);
    }

    public void CalculateHiddenGradients(NeuralLayer nextLayer)
    {
        double dow = SumDOW(nextLayer);
        m_gradient = dow * TransferDerivative(m_output);
    }

    public void UpdateInputWeights(NeuralLayer prevLayer)
    {

        // update weights in previous layer
        for (int i = 0; i != prevLayer.m_neurons.Count; i++)
        {
            Neuron curNeuron = prevLayer.m_neurons[i];
            double oldDeltaWeight = curNeuron.m_deltaWeights[m_myIndex];

            // change in weight, speed at which i learn
            double newDeltaWeight = (CONST_ETA * GetOutputValue() * m_gradient) + (CONST_ALPHA * oldDeltaWeight);

            curNeuron.m_deltaWeights[m_myIndex] = newDeltaWeight;
            curNeuron.m_outputWeights[m_myIndex] = newDeltaWeight;
        }
    }

    public double SumDOW(NeuralLayer nextLayer)
    {
        double sum = 0.0;

        // sum all contributions of errors for next layer
        for (int i = 0; i < nextLayer.m_neurons.Count - 1; i++)
        {
            sum += m_outputWeights[i] * nextLayer.m_neurons[i].m_gradient;
        }

        return sum;
    }
}
