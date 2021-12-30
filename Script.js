//training set. [input1, input2, output(s)]
var data = [
  [1,   0.5, [0]],
  [1.5, 1,   [0]],
  [1.5, 1.5, [0]],
  [2,   1.5, [0]], 

  [3, 2, [1]],
  [3,   1.5, [1]],
  [3,   1,   [1]],
  [3.5, 1,   [1]],
];

//unknown type (data we want to find)
var dataU = [
  [4,   2, "it should be 1"]
];

//asking for the size of the network
var NumOfHidLayers = 2 //prompt("The number hidden layers:");
if(NumOfHidLayers == 0) {EveryHidLayerSize = 0;} else {
var EveryHidLayerSize = 2 //prompt("The size of each hidden layer:");
};
//Weights config (bias included)
var NumOfInputLayers = data[0].length - 1;
var NumOfOutputs = data[0][data[0].length - 1].length
var NumOfWeights = (NumOfInputLayers * EveryHidLayerSize) + ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) ) + (EveryHidLayerSize * NumOfOutputs);
  if(NumOfHidLayers == 0) {NumOfWeights = NumOfInputLayers * NumOfOutputs};
var bias = [0]
for(count = 0; count < (NumOfHidLayers + 1); count++) {
  bias[count] = Math.random()
};
var w = [0]
for(count = 0; count < NumOfWeights; count++) {
  w[count] = Math.random()
};

//reload every variable in acurdents to the dataset (i: data selector)
var node = [[]];
var NN = [];
function reload(i) {

	//Neural nets config
   //************************************************************************
  
 	  //reseting all nodes to 0  
    for(count = 0; count < NumOfOutputs; count++) {NN[count] = 0};
    
    var k = 0;
    var t = 0;
    for(count = 0; count < NumOfHidLayers; count++) {node[count] = [];};
		for(count = 0; count < (NumOfHidLayers * EveryHidLayerSize); count++) {
			node[k][t] = 0;
      t++;
      if(t == EveryHidLayerSize) {
    	  k++;
    	  t = 0;
      };
		};
    
	  //First hidden layer
		k = 0
    t = 0
		for(count = 0; count < (NumOfInputLayers * EveryHidLayerSize); count++) {
      node[0][k] += data[i][t] * w[count];
      t++;
      if(t == NumOfInputLayers) {
     	 node[0][k] += bias[0];
     	 k++;
     	 t = 0;
      };
		};

	  //Rest of the hidden layers
    k = 1;
    t = 0;
    var h = 0;
    var n = 1
    n = EveryHidLayerSize;
    var NumOfCountTillNow = node[0].length;
		for(count = NumOfCountTillNow; count < ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) + NumOfCountTillNow); count++) {
      node[k][t] += node[k - 1][h] * w[count];
      h++;
      if(h == EveryHidLayerSize) {
        node[k][t] += bias[n];
     	  h = 0;
        t++
        if(t == EveryHidLayerSize) {
       	 n++;
       	 k++;
       	 t = 0; 
        }
      };
    };

		//Output layer 
    k = 0;
    t = 0;
    n = NumOfHidLayers
    NumOfCountTillNow = ( (NumOfHidLayers - 1) * Math.pow(EveryHidLayerSize, 2) + node[0].length);
    for(count = NumOfCountTillNow; count < ( (NumOfOutputs * EveryHidLayerSize) + NumOfCountTillNow); count++) {
    	NN[k] += node[NumOfHidLayers - 1][t] * w[count];
      t++;
      if(t == EveryHidLayerSize) {
     	 NN[k] += bias[n];
       n++
     	 k++;
     	 t = 0;
      };
    };
    //**In case we had no hidden layer
    if(NumOfHidLayers == 0) {
    	k = 0
    	t = 0
      n = 0
    	for(count = 0; count < (NumOfInputLayers * NumOfOutputs); count++) {
      	NN[k] += data[i][t] * w[count];
      	t++;
      	if(t == NumOfInputLayers) {
     	 		NN[k] += bias[n];
     	 		k++;
     	 		t = 0;
      	};
			};
    };
    
   //************************************************************************
    
	//Slope and stuff

		//Defining sigmoid
		function Sig(x) {return (1 / (1 + Math.pow(2.718, x))); };
    
    
    
/*
		//squared error for every individual dataset (just for fun)
var everycost = Math.pow((S - c[i]), 2);

// slope of the error and the change we will want to make to each smaller value
var everyslope = 2 * (S - c[i]);
function slopeF() {
   var clope = 0;
   for(i = 1; i < 9; i++) {
   reload();
   clope = clope + everyslope;
  };
  return clope;
};
var slope = slopeF();
var d = 5000;
var change = slope/d;

//function to Reload variables
function reload() {
  NN1 = (w[0] * x[i] + w[1] * y[i] + w[2]);
  NN2 = (w[3] * x[i] + w[4] * y[i] + w[5]);
  NN3 = (w[6] * x[i] + w[7] * y[i] + w[8]);
  NN = (w[9] * NN1 + w[10] * NN2 + w[11] * NN3 + w[12]);
  S = 1 / (1 + Math.pow(2.718, NN));
  everycost = Math.pow((S - c[i]), 2);
  everyslope = 2 * (S - c[i]);
};

//overall cost (just for fun)
function cost() {
   var kost = 0;
   for(i = 1; i < x.length; i++) {
   reload();
   kost = kost + everycost;
 };
 return kost;
};


//self explanatory
var loopcounter = 0;
var randomchoice1 = 0;
var randomchoice2 = 0;
i = 1;
var k = 1;
d = 1000
reload();

//finding the "right" value for w1, w2 and b
for (loopcounter = 0; loopcounter < 10000 ; loopcounter++) {

  reload();
  slope = slopeF();
  change = slope/d;
  i = k
   if (slope > 0) {
      if(randomchoice1 <= 12) {
        w[randomchoice1] = w[randomchoice1] + change
      }
   };

   if (slope < 0) {
      if(randomchoice2 <= 12) {
        w[randomchoice2] = w[randomchoice2] + change
      }
   };
  
  if (slope == 0) {break;}

   randomchoice1++;
   randomchoice2++;
   k++;
   if (k == 9) {k = 1};
   if (randomchoice1 == 13) {randomchoice1 = 0};
   if (randomchoice2 == 13) {randomchoice2 = 0};
};

//request for answer
i = 0;
reload();
dataU[2] = S;

var perc = Math.floor(dataU[2] * 10000)/100
if(dataU[2] < 0.7 && dataU[2] > 0.3) {alert("iduno. color variable: " + (perc / 100) + " (Blue: 0 and Red: 1)")} else {

    if(dataU[2] >= 0.7) {alert("I'm " + perc + "% sure its a Red flower");}
      
    if(dataU[2] <= 0.3) { alert("I'm " + (100 - perc) + "% sure its a Blue flower"); }
}:
*/