#include "pch.h"
#include "PID.h"

/**********************************************************************************************
 * Arduino PID Library - Version 1.2.1
 * by Brett Beauregard <br3ttb@gmail.com> brettbeauregard.com
 *
 * This Library is licensed under the MIT License
 **********************************************************************************************/


 /*Constructor (...)*********************************************************
  *    The parameters specified here are those for for which we can't set up
  *    reliable defaults, so we need to have the user set them.
  ***************************************************************************/
PID::PID(double* Input, double* Output, double* Setpoint,
    double Kp, double Ki, double Kd)
{
    myOutput = Output;
    myInput = Input;
    mySetpoint = Setpoint;

    PID::SetOutputLimits(-9999999999, 9999999999);				//default output limit corresponds to
                                                    //the arduino pwm limits

    PID::SetTunings(Kp, Ki, Kd);
}

/* Compute() **********************************************************************
 *     This, as they say, is where the magic happens.  this function should be called
 *   every time "void loop()" executes.  the function will decide for itself whether a new
 *   pid Output needs to be computed.  returns true when the output is computed,
 *   false when nothing has been done.
 **********************************************************************************/
void PID::Compute()
{
    /*Compute all the working error variables*/
    double input = *myInput;
    double error = *mySetpoint - input;
    double dInput = (input - lastInput);
    outputSum += (ki * error);

    if (outputSum > outMax) outputSum = outMax;
    else if (outputSum < outMin) outputSum = outMin;

    /*Add Proportional on Error*/
    double output  = kp * error;

    /*Compute Rest of PID Output*/
    output += outputSum - kd * dInput;

    if (output > outMax) output = outMax;
    else if (output < outMin) output = outMin;
    *myOutput = output;

    /*Remember some variables for next time*/
    lastInput = input;
}

/* SetTunings(...)*************************************************************
 * This function allows the controller's dynamic performance to be adjusted.
 * it's called automatically from the constructor, but tunings can also
 * be adjusted on the fly during normal operation
 ******************************************************************************/
void PID::SetTunings(double Kp, double Ki, double Kd)
{
    if (Kp < 0 || Ki < 0 || Kd < 0) return;

    dispKp = Kp; dispKi = Ki; dispKd = Kd;

    kp = Kp;
    ki = Ki;
    kd = Kd;
}


/* SetOutputLimits(...)****************************************************
 *     This function will be used far more often than SetInputLimits.  while
 *  the input to the controller will generally be in the 0-1023 range (which is
 *  the default already,)  the output will be a little different.  maybe they'll
 *  be doing a time window and will need 0-8000 or something.  or maybe they'll
 *  want to clamp it from 0-125.  who knows.  at any rate, that can all be done
 *  here.
 **************************************************************************/
void PID::SetOutputLimits(double Min, double Max)
{
    if (Min >= Max) return;
    outMin = Min;
    outMax = Max;

    if (*myOutput > outMax) *myOutput = outMax;
    else if (*myOutput < outMin) *myOutput = outMin;

    if (outputSum > outMax) outputSum = outMax;
    else if (outputSum < outMin) outputSum = outMin;
}

/* Initialize()****************************************************************
 *	does all the things that need to happen to ensure a bumpless transfer
 *  from manual to automatic mode.
 ******************************************************************************/
void PID::Initialize()
{
    outputSum = *myOutput;
    lastInput = *myInput;
    if (outputSum > outMax) outputSum = outMax;
    else if (outputSum < outMin) outputSum = outMin;
}

/* Status Funcions*************************************************************
 * Just because you set the Kp=-1 doesn't mean it actually happened.  these
 * functions query the internal state of the PID.  they're here for display
 * purposes.  this are the functions the PID Front-end uses for example
 ******************************************************************************/
double PID::GetKp() { return  dispKp; }
double PID::GetKi() { return  dispKi; }
double PID::GetKd() { return  dispKd; }
