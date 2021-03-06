??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
}
!FakeQuantWithMinMaxVarsPerChannel

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
%quantize_layer_1/quantize_layer_1_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%quantize_layer_1/quantize_layer_1_min
?
9quantize_layer_1/quantize_layer_1_min/Read/ReadVariableOpReadVariableOp%quantize_layer_1/quantize_layer_1_min*
_output_shapes
: *
dtype0
?
%quantize_layer_1/quantize_layer_1_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%quantize_layer_1/quantize_layer_1_max
?
9quantize_layer_1/quantize_layer_1_max/Read/ReadVariableOpReadVariableOp%quantize_layer_1/quantize_layer_1_max*
_output_shapes
: *
dtype0
?
quantize_layer_1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quantize_layer_1/optimizer_step
?
3quantize_layer_1/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer_1/optimizer_step*
_output_shapes
: *
dtype0
?
quant_reshape_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name quant_reshape_2/optimizer_step
?
2quant_reshape_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_reshape_2/optimizer_step*
_output_shapes
: *
dtype0
?
quant_conv2d_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_conv2d_2/optimizer_step
?
1quant_conv2d_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d_2/optimizer_step*
_output_shapes
: *
dtype0
?
quant_conv2d_2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_2/kernel_min
?
-quant_conv2d_2/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_min*
_output_shapes
:*
dtype0
?
quant_conv2d_2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namequant_conv2d_2/kernel_max
?
-quant_conv2d_2/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d_2/kernel_max*
_output_shapes
:*
dtype0
?
"quant_conv2d_2/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_min
?
6quant_conv2d_2/post_activation_min/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_min*
_output_shapes
: *
dtype0
?
"quant_conv2d_2/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_conv2d_2/post_activation_max
?
6quant_conv2d_2/post_activation_max/Read/ReadVariableOpReadVariableOp"quant_conv2d_2/post_activation_max*
_output_shapes
: *
dtype0
?
$quant_max_pooling2d_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$quant_max_pooling2d_2/optimizer_step
?
8quant_max_pooling2d_2/optimizer_step/Read/ReadVariableOpReadVariableOp$quant_max_pooling2d_2/optimizer_step*
_output_shapes
: *
dtype0
?
quant_flatten_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name quant_flatten_2/optimizer_step
?
2quant_flatten_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_flatten_2/optimizer_step*
_output_shapes
: *
dtype0
?
quant_dense_2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dense_2/optimizer_step
?
0quant_dense_2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense_2/optimizer_step*
_output_shapes
: *
dtype0
?
quant_dense_2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_2/kernel_min
}
,quant_dense_2/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense_2/kernel_min*
_output_shapes
: *
dtype0
?
quant_dense_2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_dense_2/kernel_max
}
,quant_dense_2/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense_2/kernel_max*
_output_shapes
: *
dtype0
?
!quant_dense_2/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_2/post_activation_min
?
5quant_dense_2/post_activation_min/Read/ReadVariableOpReadVariableOp!quant_dense_2/post_activation_min*
_output_shapes
: *
dtype0
?
!quant_dense_2/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_dense_2/post_activation_max
?
5quant_dense_2/post_activation_max/Read/ReadVariableOpReadVariableOp!quant_dense_2/post_activation_max*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?
*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	?
*
dtype0

NoOpNoOp
?K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?J
value?JB?J B?J
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
?
quantize_layer_1_min
quantize_layer_1_max
quantizer_vars
optimizer_step
regularization_losses
trainable_variables
	variables
	keras_api
?
	layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers
regularization_losses
trainable_variables
	variables
	keras_api
?
	layer
optimizer_step
 _weight_vars
!
kernel_min
"
kernel_max
#_quantize_activations
$post_activation_min
%post_activation_max
&_output_quantizers
'regularization_losses
(trainable_variables
)	variables
*	keras_api
?
	+layer
,optimizer_step
-_weight_vars
._quantize_activations
/_output_quantizers
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?
	4layer
5optimizer_step
6_weight_vars
7_quantize_activations
8_output_quantizers
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?
	=layer
>optimizer_step
?_weight_vars
@
kernel_min
A
kernel_max
B_quantize_activations
Cpost_activation_min
Dpost_activation_max
E_output_quantizers
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_rateOm?Pm?Qm?Rm?Ov?Pv?Qv?Rv?
 

O0
P1
Q2
R3
?
0
1
2
3
O4
P5
6
!7
"8
$9
%10
,11
512
Q13
R14
>15
@16
A17
C18
D19
?
regularization_losses
Snon_trainable_variables
	trainable_variables

	variables
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics

Wlayers
 
}
VARIABLE_VALUE%quantize_layer_1/quantize_layer_1_minDlayer_with_weights-0/quantize_layer_1_min/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE%quantize_layer_1/quantize_layer_1_maxDlayer_with_weights-0/quantize_layer_1_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
sq
VARIABLE_VALUEquantize_layer_1/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
?
regularization_losses
Xnon_trainable_variables
trainable_variables
	variables
Ymetrics
Zlayer_regularization_losses
[layer_metrics

\layers
R
]regularization_losses
^trainable_variables
_	variables
`	keras_api
rp
VARIABLE_VALUEquant_reshape_2/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 

0
?
regularization_losses
anon_trainable_variables
trainable_variables
	variables
bmetrics
clayer_regularization_losses
dlayer_metrics

elayers
h

Pkernel
Obias
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
qo
VARIABLE_VALUEquant_conv2d_2/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

j0
ig
VARIABLE_VALUEquant_conv2d_2/kernel_min:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_conv2d_2/kernel_max:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
{y
VARIABLE_VALUE"quant_conv2d_2/post_activation_minClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"quant_conv2d_2/post_activation_maxClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
 

O0
P1
1
O0
P1
2
!3
"4
$5
%6
?
'regularization_losses
knon_trainable_variables
(trainable_variables
)	variables
lmetrics
mlayer_regularization_losses
nlayer_metrics

olayers
R
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
xv
VARIABLE_VALUE$quant_max_pooling2d_2/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 

,0
?
0regularization_losses
tnon_trainable_variables
1trainable_variables
2	variables
umetrics
vlayer_regularization_losses
wlayer_metrics

xlayers
R
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
rp
VARIABLE_VALUEquant_flatten_2/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 

50
?
9regularization_losses
}non_trainable_variables
:trainable_variables
;	variables
~metrics
layer_regularization_losses
?layer_metrics
?layers
l

Rkernel
Qbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
pn
VARIABLE_VALUEquant_dense_2/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

?0
hf
VARIABLE_VALUEquant_dense_2/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_dense_2/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
zx
VARIABLE_VALUE!quant_dense_2/post_activation_minClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!quant_dense_2/post_activation_maxClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
 

Q0
R1
1
Q0
R1
>2
@3
A4
C5
D6
?
Fregularization_losses
?non_trainable_variables
Gtrainable_variables
H	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_2/bias0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_2/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_2/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_2/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
v
0
1
2
3
4
!5
"6
$7
%8
,9
510
>11
@12
A13
C14
D15

?0
?1
 
 
*
0
1
2
3
4
5

0
1
2
 
 
 
 
 
 
 
?
]regularization_losses
?non_trainable_variables
^trainable_variables
_	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

0
 
 
 

0
 

O0

O0
?
fregularization_losses
?non_trainable_variables
gtrainable_variables
h	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

P0
?2
#
0
!1
"2
$3
%4
 
 
 

0
 
 
 
?
pregularization_losses
?non_trainable_variables
qtrainable_variables
r	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

,0
 
 
 

+0
 
 
 
?
yregularization_losses
?non_trainable_variables
ztrainable_variables
{	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

50
 
 
 

40
 

Q0

Q0
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers

R0
?2
#
>0
@1
A2
C3
D4
 
 
 

=0
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 

!min_var
"max_var
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

@min_var
Amax_var
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
vt
VARIABLE_VALUEAdam/conv2d_2/bias/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_2/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_2/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_2/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_2/bias/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_2/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_2/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_2/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_3Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3%quantize_layer_1/quantize_layer_1_min%quantize_layer_1/quantize_layer_1_maxconv2d_2/kernelquant_conv2d_2/kernel_minquant_conv2d_2/kernel_maxconv2d_2/bias"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_maxdense_2/kernelquant_dense_2/kernel_minquant_dense_2/kernel_maxdense_2/bias!quant_dense_2/post_activation_min!quant_dense_2/post_activation_max*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_69108
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9quantize_layer_1/quantize_layer_1_min/Read/ReadVariableOp9quantize_layer_1/quantize_layer_1_max/Read/ReadVariableOp3quantize_layer_1/optimizer_step/Read/ReadVariableOp2quant_reshape_2/optimizer_step/Read/ReadVariableOp1quant_conv2d_2/optimizer_step/Read/ReadVariableOp-quant_conv2d_2/kernel_min/Read/ReadVariableOp-quant_conv2d_2/kernel_max/Read/ReadVariableOp6quant_conv2d_2/post_activation_min/Read/ReadVariableOp6quant_conv2d_2/post_activation_max/Read/ReadVariableOp8quant_max_pooling2d_2/optimizer_step/Read/ReadVariableOp2quant_flatten_2/optimizer_step/Read/ReadVariableOp0quant_dense_2/optimizer_step/Read/ReadVariableOp,quant_dense_2/kernel_min/Read/ReadVariableOp,quant_dense_2/kernel_max/Read/ReadVariableOp5quant_dense_2/post_activation_min/Read/ReadVariableOp5quant_dense_2/post_activation_max/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_69846
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%quantize_layer_1/quantize_layer_1_min%quantize_layer_1/quantize_layer_1_maxquantize_layer_1/optimizer_stepquant_reshape_2/optimizer_stepquant_conv2d_2/optimizer_stepquant_conv2d_2/kernel_minquant_conv2d_2/kernel_max"quant_conv2d_2/post_activation_min"quant_conv2d_2/post_activation_max$quant_max_pooling2d_2/optimizer_stepquant_flatten_2/optimizer_stepquant_dense_2/optimizer_stepquant_dense_2/kernel_minquant_dense_2/kernel_max!quant_dense_2/post_activation_min!quant_dense_2/post_activation_max	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_2/biasconv2d_2/kerneldense_2/biasdense_2/kerneltotalcounttotal_1count_1Adam/conv2d_2/bias/mAdam/conv2d_2/kernel/mAdam/dense_2/bias/mAdam/dense_2/kernel/mAdam/conv2d_2/bias/vAdam/conv2d_2/kernel/vAdam/dense_2/bias/vAdam/dense_2/kernel/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_69967??
ʱ
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69360

inputsL
Bquantize_layer_1_allvaluesquantize_minimum_readvariableop_resource: L
Bquantize_layer_1_allvaluesquantize_maximum_readvariableop_resource: X
>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource:B
4quant_conv2d_2_lastvaluequant_assignminlast_resource:B
4quant_conv2d_2_lastvaluequant_assignmaxlast_resource:<
.quant_conv2d_2_biasadd_readvariableop_resource:O
Equant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource: O
Equant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource: L
9quant_dense_2_lastvaluequant_rank_readvariableop_resource:	?
=
3quant_dense_2_lastvaluequant_assignminlast_resource: =
3quant_dense_2_lastvaluequant_assignmaxlast_resource: ;
-quant_dense_2_biasadd_readvariableop_resource:
N
Dquant_dense_2_movingavgquantize_assignminema_readvariableop_resource: N
Dquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resource: 
identity??%quant_conv2d_2/BiasAdd/ReadVariableOp?+quant_conv2d_2/LastValueQuant/AssignMaxLast?+quant_conv2d_2/LastValueQuant/AssignMinLast?5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp?5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp?Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp?Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?$quant_dense_2/BiasAdd/ReadVariableOp?*quant_dense_2/LastValueQuant/AssignMaxLast?*quant_dense_2/LastValueQuant/AssignMinLast?4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp?4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp?Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp?Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?4quantize_layer_1/AllValuesQuantize/AssignMaxAllValue?4quantize_layer_1/AllValuesQuantize/AssignMinAllValue?Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?9quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp?9quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp?
(quantize_layer_1/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(quantize_layer_1/AllValuesQuantize/Const?
+quantize_layer_1/AllValuesQuantize/BatchMinMininputs1quantize_layer_1/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2-
+quantize_layer_1/AllValuesQuantize/BatchMin?
*quantize_layer_1/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*quantize_layer_1/AllValuesQuantize/Const_1?
+quantize_layer_1/AllValuesQuantize/BatchMaxMaxinputs3quantize_layer_1/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2-
+quantize_layer_1/AllValuesQuantize/BatchMax?
9quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOpBquantize_layer_1_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02;
9quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp?
*quantize_layer_1/AllValuesQuantize/MinimumMinimumAquantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp:value:04quantize_layer_1/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2,
*quantize_layer_1/AllValuesQuantize/Minimum?
.quantize_layer_1/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.quantize_layer_1/AllValuesQuantize/Minimum_1/y?
,quantize_layer_1/AllValuesQuantize/Minimum_1Minimum.quantize_layer_1/AllValuesQuantize/Minimum:z:07quantize_layer_1/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2.
,quantize_layer_1/AllValuesQuantize/Minimum_1?
9quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOpBquantize_layer_1_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02;
9quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp?
*quantize_layer_1/AllValuesQuantize/MaximumMaximumAquantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp:value:04quantize_layer_1/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2,
*quantize_layer_1/AllValuesQuantize/Maximum?
.quantize_layer_1/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.quantize_layer_1/AllValuesQuantize/Maximum_1/y?
,quantize_layer_1/AllValuesQuantize/Maximum_1Maximum.quantize_layer_1/AllValuesQuantize/Maximum:z:07quantize_layer_1/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2.
,quantize_layer_1/AllValuesQuantize/Maximum_1?
4quantize_layer_1/AllValuesQuantize/AssignMinAllValueAssignVariableOpBquantize_layer_1_allvaluesquantize_minimum_readvariableop_resource0quantize_layer_1/AllValuesQuantize/Minimum_1:z:0:^quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype026
4quantize_layer_1/AllValuesQuantize/AssignMinAllValue?
4quantize_layer_1/AllValuesQuantize/AssignMaxAllValueAssignVariableOpBquantize_layer_1_allvaluesquantize_maximum_readvariableop_resource0quantize_layer_1/AllValuesQuantize/Maximum_1:z:0:^quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype026
4quantize_layer_1/AllValuesQuantize/AssignMaxAllValue?
Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquantize_layer_1_allvaluesquantize_minimum_readvariableop_resource5^quantize_layer_1/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02K
Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquantize_layer_1_allvaluesquantize_maximum_readvariableop_resource5^quantize_layer_1/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02M
Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
:quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsQquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Squantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2<
:quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars?
quant_reshape_2/ShapeShapeDquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*
_output_shapes
:2
quant_reshape_2/Shape?
#quant_reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#quant_reshape_2/strided_slice/stack?
%quant_reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%quant_reshape_2/strided_slice/stack_1?
%quant_reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%quant_reshape_2/strided_slice/stack_2?
quant_reshape_2/strided_sliceStridedSlicequant_reshape_2/Shape:output:0,quant_reshape_2/strided_slice/stack:output:0.quant_reshape_2/strided_slice/stack_1:output:0.quant_reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quant_reshape_2/strided_slice?
quant_reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
quant_reshape_2/Reshape/shape/1?
quant_reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
quant_reshape_2/Reshape/shape/2?
quant_reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
quant_reshape_2/Reshape/shape/3?
quant_reshape_2/Reshape/shapePack&quant_reshape_2/strided_slice:output:0(quant_reshape_2/Reshape/shape/1:output:0(quant_reshape_2/Reshape/shape/2:output:0(quant_reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
quant_reshape_2/Reshape/shape?
quant_reshape_2/ReshapeReshapeDquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0&quant_reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
quant_reshape_2/Reshape?
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp?
8quant_conv2d_2/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_2/LastValueQuant/BatchMin/reduction_indices?
&quant_conv2d_2/LastValueQuant/BatchMinMin=quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_2/LastValueQuant/BatchMin?
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype027
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp?
8quant_conv2d_2/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_conv2d_2/LastValueQuant/BatchMax/reduction_indices?
&quant_conv2d_2/LastValueQuant/BatchMaxMax=quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_conv2d_2/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2(
&quant_conv2d_2/LastValueQuant/BatchMax?
'quant_conv2d_2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'quant_conv2d_2/LastValueQuant/truediv/y?
%quant_conv2d_2/LastValueQuant/truedivRealDiv/quant_conv2d_2/LastValueQuant/BatchMax:output:00quant_conv2d_2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2'
%quant_conv2d_2/LastValueQuant/truediv?
%quant_conv2d_2/LastValueQuant/MinimumMinimum/quant_conv2d_2/LastValueQuant/BatchMin:output:0)quant_conv2d_2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_2/LastValueQuant/Minimum?
#quant_conv2d_2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#quant_conv2d_2/LastValueQuant/mul/y?
!quant_conv2d_2/LastValueQuant/mulMul/quant_conv2d_2/LastValueQuant/BatchMin:output:0,quant_conv2d_2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2#
!quant_conv2d_2/LastValueQuant/mul?
%quant_conv2d_2/LastValueQuant/MaximumMaximum/quant_conv2d_2/LastValueQuant/BatchMax:output:0%quant_conv2d_2/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2'
%quant_conv2d_2/LastValueQuant/Maximum?
+quant_conv2d_2/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource)quant_conv2d_2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_2/LastValueQuant/AssignMinLast?
+quant_conv2d_2/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource)quant_conv2d_2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_conv2d_2/LastValueQuant/AssignMaxLast?
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource,^quant_conv2d_2/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource,^quant_conv2d_2/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
quant_conv2d_2/Conv2DConv2D quant_reshape_2/Reshape:output:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
quant_conv2d_2/Conv2D?
%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_2/BiasAdd/ReadVariableOp?
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
quant_conv2d_2/BiasAdd?
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
quant_conv2d_2/Relu?
&quant_conv2d_2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d_2/MovingAvgQuantize/Const?
)quant_conv2d_2/MovingAvgQuantize/BatchMinMin!quant_conv2d_2/Relu:activations:0/quant_conv2d_2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_2/MovingAvgQuantize/BatchMin?
(quant_conv2d_2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quant_conv2d_2/MovingAvgQuantize/Const_1?
)quant_conv2d_2/MovingAvgQuantize/BatchMaxMax!quant_conv2d_2/Relu:activations:01quant_conv2d_2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quant_conv2d_2/MovingAvgQuantize/BatchMax?
*quant_conv2d_2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_2/MovingAvgQuantize/Minimum/y?
(quant_conv2d_2/MovingAvgQuantize/MinimumMinimum2quant_conv2d_2/MovingAvgQuantize/BatchMin:output:03quant_conv2d_2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_2/MovingAvgQuantize/Minimum?
*quant_conv2d_2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*quant_conv2d_2/MovingAvgQuantize/Maximum/y?
(quant_conv2d_2/MovingAvgQuantize/MaximumMaximum2quant_conv2d_2/MovingAvgQuantize/BatchMax:output:03quant_conv2d_2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2*
(quant_conv2d_2/MovingAvgQuantize/Maximum?
3quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decay?
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp?
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/sub?
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mul?
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMinEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
3quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decay?
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02>
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/subSubDquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0,quant_conv2d_2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/sub?
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mulMul5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/sub:z:0<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 23
1quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mul?
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resource5quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/mul:z:0=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02C
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpEquant_conv2d_2_movingavgquantize_assignminema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpEquant_conv2d_2_movingavgquantize_assignmaxema_readvariableop_resourceB^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02K
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2:
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars?
quant_max_pooling2d_2/MaxPoolMaxPoolBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
quant_max_pooling2d_2/MaxPool
quant_flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
quant_flatten_2/Const?
quant_flatten_2/ReshapeReshape&quant_max_pooling2d_2/MaxPool:output:0quant_flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
quant_flatten_2/Reshape?
0quant_dense_2/LastValueQuant/Rank/ReadVariableOpReadVariableOp9quant_dense_2_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype022
0quant_dense_2/LastValueQuant/Rank/ReadVariableOp?
!quant_dense_2/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2#
!quant_dense_2/LastValueQuant/Rank?
(quant_dense_2/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(quant_dense_2/LastValueQuant/range/start?
(quant_dense_2/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(quant_dense_2/LastValueQuant/range/delta?
"quant_dense_2/LastValueQuant/rangeRange1quant_dense_2/LastValueQuant/range/start:output:0*quant_dense_2/LastValueQuant/Rank:output:01quant_dense_2/LastValueQuant/range/delta:output:0*
_output_shapes
:2$
"quant_dense_2/LastValueQuant/range?
4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp9quant_dense_2_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype026
4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp?
%quant_dense_2/LastValueQuant/BatchMinMin<quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp:value:0+quant_dense_2/LastValueQuant/range:output:0*
T0*
_output_shapes
: 2'
%quant_dense_2/LastValueQuant/BatchMin?
2quant_dense_2/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp9quant_dense_2_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype024
2quant_dense_2/LastValueQuant/Rank_1/ReadVariableOp?
#quant_dense_2/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#quant_dense_2/LastValueQuant/Rank_1?
*quant_dense_2/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*quant_dense_2/LastValueQuant/range_1/start?
*quant_dense_2/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*quant_dense_2/LastValueQuant/range_1/delta?
$quant_dense_2/LastValueQuant/range_1Range3quant_dense_2/LastValueQuant/range_1/start:output:0,quant_dense_2/LastValueQuant/Rank_1:output:03quant_dense_2/LastValueQuant/range_1/delta:output:0*
_output_shapes
:2&
$quant_dense_2/LastValueQuant/range_1?
4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp9quant_dense_2_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype026
4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp?
%quant_dense_2/LastValueQuant/BatchMaxMax<quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp:value:0-quant_dense_2/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2'
%quant_dense_2/LastValueQuant/BatchMax?
&quant_dense_2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&quant_dense_2/LastValueQuant/truediv/y?
$quant_dense_2/LastValueQuant/truedivRealDiv.quant_dense_2/LastValueQuant/BatchMax:output:0/quant_dense_2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2&
$quant_dense_2/LastValueQuant/truediv?
$quant_dense_2/LastValueQuant/MinimumMinimum.quant_dense_2/LastValueQuant/BatchMin:output:0(quant_dense_2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2&
$quant_dense_2/LastValueQuant/Minimum?
"quant_dense_2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"quant_dense_2/LastValueQuant/mul/y?
 quant_dense_2/LastValueQuant/mulMul.quant_dense_2/LastValueQuant/BatchMin:output:0+quant_dense_2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2"
 quant_dense_2/LastValueQuant/mul?
$quant_dense_2/LastValueQuant/MaximumMaximum.quant_dense_2/LastValueQuant/BatchMax:output:0$quant_dense_2/LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2&
$quant_dense_2/LastValueQuant/Maximum?
*quant_dense_2/LastValueQuant/AssignMinLastAssignVariableOp3quant_dense_2_lastvaluequant_assignminlast_resource(quant_dense_2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02,
*quant_dense_2/LastValueQuant/AssignMinLast?
*quant_dense_2/LastValueQuant/AssignMaxLastAssignVariableOp3quant_dense_2_lastvaluequant_assignmaxlast_resource(quant_dense_2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02,
*quant_dense_2/LastValueQuant/AssignMaxLast?
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9quant_dense_2_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02E
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp3quant_dense_2_lastvaluequant_assignminlast_resource+^quant_dense_2/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype02G
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp3quant_dense_2_lastvaluequant_assignmaxlast_resource+^quant_dense_2/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype02G
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
4quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(26
4quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars?
quant_dense_2/MatMulMatMul quant_flatten_2/Reshape:output:0>quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2
quant_dense_2/MatMul?
$quant_dense_2/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02&
$quant_dense_2/BiasAdd/ReadVariableOp?
quant_dense_2/BiasAddBiasAddquant_dense_2/MatMul:product:0,quant_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
quant_dense_2/BiasAdd?
%quant_dense_2/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%quant_dense_2/MovingAvgQuantize/Const?
(quant_dense_2/MovingAvgQuantize/BatchMinMinquant_dense_2/BiasAdd:output:0.quant_dense_2/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2*
(quant_dense_2/MovingAvgQuantize/BatchMin?
'quant_dense_2/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'quant_dense_2/MovingAvgQuantize/Const_1?
(quant_dense_2/MovingAvgQuantize/BatchMaxMaxquant_dense_2/BiasAdd:output:00quant_dense_2/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2*
(quant_dense_2/MovingAvgQuantize/BatchMax?
)quant_dense_2/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_dense_2/MovingAvgQuantize/Minimum/y?
'quant_dense_2/MovingAvgQuantize/MinimumMinimum1quant_dense_2/MovingAvgQuantize/BatchMin:output:02quant_dense_2/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2)
'quant_dense_2/MovingAvgQuantize/Minimum?
)quant_dense_2/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_dense_2/MovingAvgQuantize/Maximum/y?
'quant_dense_2/MovingAvgQuantize/MaximumMaximum1quant_dense_2/MovingAvgQuantize/BatchMax:output:02quant_dense_2/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2)
'quant_dense_2/MovingAvgQuantize/Maximum?
2quant_dense_2/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2quant_dense_2/MovingAvgQuantize/AssignMinEma/decay?
;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_dense_2_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp?
0quant_dense_2/MovingAvgQuantize/AssignMinEma/subSubCquant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_dense_2/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 22
0quant_dense_2/MovingAvgQuantize/AssignMinEma/sub?
0quant_dense_2/MovingAvgQuantize/AssignMinEma/mulMul4quant_dense_2/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_dense_2/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_dense_2/MovingAvgQuantize/AssignMinEma/mul?
@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_2_movingavgquantize_assignminema_readvariableop_resource4quant_dense_2/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
2quant_dense_2/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2quant_dense_2/MovingAvgQuantize/AssignMaxEma/decay?
;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
0quant_dense_2/MovingAvgQuantize/AssignMaxEma/subSubCquant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_dense_2/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 22
0quant_dense_2/MovingAvgQuantize/AssignMaxEma/sub?
0quant_dense_2/MovingAvgQuantize/AssignMaxEma/mulMul4quant_dense_2/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_dense_2/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_dense_2/MovingAvgQuantize/AssignMaxEma/mul?
@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resource4quant_dense_2/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_dense_2_movingavgquantize_assignminema_readvariableop_resourceA^quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02H
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_dense_2_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02J
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_2/BiasAdd:output:0Nquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
29
7quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentityAquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp&^quant_conv2d_2/BiasAdd/ReadVariableOp,^quant_conv2d_2/LastValueQuant/AssignMaxLast,^quant_conv2d_2/LastValueQuant/AssignMinLast6^quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp6^quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2B^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpB^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp=^quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpH^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_2/BiasAdd/ReadVariableOp+^quant_dense_2/LastValueQuant/AssignMaxLast+^quant_dense_2/LastValueQuant/AssignMinLast5^quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp5^quant_dense_2/LastValueQuant/BatchMin/ReadVariableOpD^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2A^quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp<^quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOpA^quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp<^quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOpG^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_15^quantize_layer_1/AllValuesQuantize/AssignMaxAllValue5^quantize_layer_1/AllValuesQuantize/AssignMinAllValueJ^quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpL^quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:^quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp:^quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2Z
+quant_conv2d_2/LastValueQuant/AssignMaxLast+quant_conv2d_2/LastValueQuant/AssignMaxLast2Z
+quant_conv2d_2/LastValueQuant/AssignMinLast+quant_conv2d_2/LastValueQuant/AssignMinLast2n
5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMax/ReadVariableOp2n
5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp5quant_conv2d_2/LastValueQuant/BatchMin/ReadVariableOp2?
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22?
Aquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2?
Aquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAquant_conv2d_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2|
<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp<quant_conv2d_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2?
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_2/BiasAdd/ReadVariableOp$quant_dense_2/BiasAdd/ReadVariableOp2X
*quant_dense_2/LastValueQuant/AssignMaxLast*quant_dense_2/LastValueQuant/AssignMaxLast2X
*quant_dense_2/LastValueQuant/AssignMinLast*quant_dense_2/LastValueQuant/AssignMinLast2l
4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp4quant_dense_2/LastValueQuant/BatchMax/ReadVariableOp2l
4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp4quant_dense_2/LastValueQuant/BatchMin/ReadVariableOp2?
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22?
@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_dense_2/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2z
;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp;quant_dense_2/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2?
@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_dense_2/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2z
;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp;quant_dense_2/MovingAvgQuantize/AssignMinEma/ReadVariableOp2?
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12l
4quantize_layer_1/AllValuesQuantize/AssignMaxAllValue4quantize_layer_1/AllValuesQuantize/AssignMaxAllValue2l
4quantize_layer_1/AllValuesQuantize/AssignMinAllValue4quantize_layer_1/AllValuesQuantize/AssignMinAllValue2?
Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpIquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12v
9quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp9quantize_layer_1/AllValuesQuantize/Maximum/ReadVariableOp2v
9quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp9quantize_layer_1/AllValuesQuantize/Minimum/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Q
5__inference_quant_max_pooling2d_2_layer_call_fn_69555

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_685052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_68513

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69067
input_3 
quantize_layer_1_69032:  
quantize_layer_1_69034: .
quant_conv2d_2_69038:"
quant_conv2d_2_69040:"
quant_conv2d_2_69042:"
quant_conv2d_2_69044:
quant_conv2d_2_69046: 
quant_conv2d_2_69048: &
quant_dense_2_69053:	?

quant_dense_2_69055: 
quant_dense_2_69057: !
quant_dense_2_69059:

quant_dense_2_69061: 
quant_dense_2_69063: 
identity??&quant_conv2d_2/StatefulPartitionedCall?%quant_dense_2/StatefulPartitionedCall?(quantize_layer_1/StatefulPartitionedCall?
(quantize_layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_3quantize_layer_1_69032quantize_layer_1_69034*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_688472*
(quantize_layer_1/StatefulPartitionedCall?
quant_reshape_2/PartitionedCallPartitionedCall1quantize_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_688112!
quant_reshape_2/PartitionedCall?
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(quant_reshape_2/PartitionedCall:output:0quant_conv2d_2_69038quant_conv2d_2_69040quant_conv2d_2_69042quant_conv2d_2_69044quant_conv2d_2_69046quant_conv2d_2_69048*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_687742(
&quant_conv2d_2/StatefulPartitionedCall?
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_687022'
%quant_max_pooling2d_2/PartitionedCall?
quant_flatten_2/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_686862!
quant_flatten_2/PartitionedCall?
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall(quant_flatten_2/PartitionedCall:output:0quant_dense_2_69053quant_dense_2_69055quant_dense_2_69057quant_dense_2_69059quant_dense_2_69061quant_dense_2_69063*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_686572'
%quant_dense_2/StatefulPartitionedCall?
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^quant_conv2d_2/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall)^quantize_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2T
(quantize_layer_1/StatefulPartitionedCall(quantize_layer_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?'
?
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_69501

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_69141

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_685502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_68702

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_69592

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?^
?	
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_68774

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indices?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMin?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indices?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?^
?	
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_69550

inputsI
/lastvaluequant_batchmin_readvariableop_resource:3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indices?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMin?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indices?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
.__inference_quant_conv2d_2_layer_call_fn_69463

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_684862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_69446

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_68535

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:
K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?
*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_68811

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_quant_flatten_2_layer_call_fn_69575

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_685132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_68991
input_3
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_689272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
0__inference_quantize_layer_1_layer_call_fn_69378

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_688472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_68550

inputs 
quantize_layer_1_68444:  
quantize_layer_1_68446: .
quant_conv2d_2_68487:"
quant_conv2d_2_68489:"
quant_conv2d_2_68491:"
quant_conv2d_2_68493:
quant_conv2d_2_68495: 
quant_conv2d_2_68497: &
quant_dense_2_68536:	?

quant_dense_2_68538: 
quant_dense_2_68540: !
quant_dense_2_68542:

quant_dense_2_68544: 
quant_dense_2_68546: 
identity??&quant_conv2d_2/StatefulPartitionedCall?%quant_dense_2/StatefulPartitionedCall?(quantize_layer_1/StatefulPartitionedCall?
(quantize_layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_1_68444quantize_layer_1_68446*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_684432*
(quantize_layer_1/StatefulPartitionedCall?
quant_reshape_2/PartitionedCallPartitionedCall1quantize_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_684632!
quant_reshape_2/PartitionedCall?
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(quant_reshape_2/PartitionedCall:output:0quant_conv2d_2_68487quant_conv2d_2_68489quant_conv2d_2_68491quant_conv2d_2_68493quant_conv2d_2_68495quant_conv2d_2_68497*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_684862(
&quant_conv2d_2/StatefulPartitionedCall?
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_685052'
%quant_max_pooling2d_2/PartitionedCall?
quant_flatten_2/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_685132!
quant_flatten_2/PartitionedCall?
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall(quant_flatten_2/PartitionedCall:output:0quant_dense_2_68536quant_dense_2_68538quant_dense_2_68540quant_dense_2_68542quant_dense_2_68544quant_dense_2_68546*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_685352'
%quant_dense_2/StatefulPartitionedCall?
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^quant_conv2d_2/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall)^quantize_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2T
(quantize_layer_1/StatefulPartitionedCall(quantize_layer_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_68581
input_3
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_685502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?	
?
.__inference_quant_conv2d_2_layer_call_fn_69480

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_687742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_68414

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_quant_max_pooling2d_2_layer_call_fn_69560

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_687022
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_69387

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_68847

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity??#AllValuesQuantize/AssignMaxAllValue?#AllValuesQuantize/AssignMinAllValue?8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?(AllValuesQuantize/Maximum/ReadVariableOp?(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
AllValuesQuantize/Const?
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin?
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
AllValuesQuantize/Const_1?
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMax?
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum?
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y?
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1?
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOp?
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum?
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y?
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1?
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue?
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_69712

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_68405
input_3i
_sequential_2_quantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: k
asequential_2_quantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: ~
dsequential_2_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:t
fsequential_2_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:t
fsequential_2_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:I
;sequential_2_quant_conv2d_2_biasadd_readvariableop_resource:g
]sequential_2_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: i
_sequential_2_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: l
Ysequential_2_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?
e
[sequential_2_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: e
[sequential_2_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: H
:sequential_2_quant_dense_2_biasadd_readvariableop_resource:
f
\sequential_2_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: h
^sequential_2_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??2sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOp?[sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?Tsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Vsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?1sequential_2/quant_dense_2/BiasAdd/ReadVariableOp?Psequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?Ssequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Usequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?Vsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Xsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Vsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp_sequential_2_quantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02X
Vsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Xsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpasequential_2_quantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Z
Xsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Gsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_3^sequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0`sequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2I
Gsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars?
"sequential_2/quant_reshape_2/ShapeShapeQsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*
_output_shapes
:2$
"sequential_2/quant_reshape_2/Shape?
0sequential_2/quant_reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_2/quant_reshape_2/strided_slice/stack?
2sequential_2/quant_reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_2/quant_reshape_2/strided_slice/stack_1?
2sequential_2/quant_reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_2/quant_reshape_2/strided_slice/stack_2?
*sequential_2/quant_reshape_2/strided_sliceStridedSlice+sequential_2/quant_reshape_2/Shape:output:09sequential_2/quant_reshape_2/strided_slice/stack:output:0;sequential_2/quant_reshape_2/strided_slice/stack_1:output:0;sequential_2/quant_reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_2/quant_reshape_2/strided_slice?
,sequential_2/quant_reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_2/quant_reshape_2/Reshape/shape/1?
,sequential_2/quant_reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_2/quant_reshape_2/Reshape/shape/2?
,sequential_2/quant_reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_2/quant_reshape_2/Reshape/shape/3?
*sequential_2/quant_reshape_2/Reshape/shapePack3sequential_2/quant_reshape_2/strided_slice:output:05sequential_2/quant_reshape_2/Reshape/shape/1:output:05sequential_2/quant_reshape_2/Reshape/shape/2:output:05sequential_2/quant_reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2,
*sequential_2/quant_reshape_2/Reshape/shape?
$sequential_2/quant_reshape_2/ReshapeReshapeQsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:03sequential_2/quant_reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2&
$sequential_2/quant_reshape_2/Reshape?
[sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpdsequential_2_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02]
[sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpfsequential_2_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02_
]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpfsequential_2_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02_
]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
Lsequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelcsequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0esequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0esequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2N
Lsequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
"sequential_2/quant_conv2d_2/Conv2DConv2D-sequential_2/quant_reshape_2/Reshape:output:0Vsequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2$
"sequential_2/quant_conv2d_2/Conv2D?
2sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp;sequential_2_quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOp?
#sequential_2/quant_conv2d_2/BiasAddBiasAdd+sequential_2/quant_conv2d_2/Conv2D:output:0:sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2%
#sequential_2/quant_conv2d_2/BiasAdd?
 sequential_2/quant_conv2d_2/ReluRelu,sequential_2/quant_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 sequential_2/quant_conv2d_2/Relu?
Tsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp]sequential_2_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02V
Tsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Vsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp_sequential_2_quant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02X
Vsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Esequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars.sequential_2/quant_conv2d_2/Relu:activations:0\sequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0^sequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2G
Esequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars?
*sequential_2/quant_max_pooling2d_2/MaxPoolMaxPoolOsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2,
*sequential_2/quant_max_pooling2d_2/MaxPool?
"sequential_2/quant_flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2$
"sequential_2/quant_flatten_2/Const?
$sequential_2/quant_flatten_2/ReshapeReshape3sequential_2/quant_max_pooling2d_2/MaxPool:output:0+sequential_2/quant_flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2&
$sequential_2/quant_flatten_2/Reshape?
Psequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpYsequential_2_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?
*
dtype02R
Psequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp[sequential_2_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02T
Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp[sequential_2_quant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype02T
Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
Asequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsXsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Zsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Zsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(2C
Asequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars?
!sequential_2/quant_dense_2/MatMulMatMul-sequential_2/quant_flatten_2/Reshape:output:0Ksequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2#
!sequential_2/quant_dense_2/MatMul?
1sequential_2/quant_dense_2/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_quant_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1sequential_2/quant_dense_2/BiasAdd/ReadVariableOp?
"sequential_2/quant_dense_2/BiasAddBiasAdd+sequential_2/quant_dense_2/MatMul:product:09sequential_2/quant_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2$
"sequential_2/quant_dense_2/BiasAdd?
Ssequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\sequential_2_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02U
Ssequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Usequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^sequential_2_quant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Usequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Dsequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars+sequential_2/quant_dense_2/BiasAdd:output:0[sequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]sequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
2F
Dsequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentityNsequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?	
NoOpNoOp3^sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOp\^sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp^^sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1^^sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2U^sequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpW^sequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12^sequential_2/quant_dense_2/BiasAdd/ReadVariableOpQ^sequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpS^sequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1S^sequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2T^sequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpV^sequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1W^sequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpY^sequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2h
2sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOp2sequential_2/quant_conv2d_2/BiasAdd/ReadVariableOp2?
[sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp[sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2]sequential_2/quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22?
Tsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpTsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Vsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Vsequential_2/quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12f
1sequential_2/quant_dense_2/BiasAdd/ReadVariableOp1sequential_2/quant_dense_2/BiasAdd/ReadVariableOp2?
Psequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpPsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2?
Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Rsequential_2/quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22?
Ssequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpSsequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Usequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Usequential_2/quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Vsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpVsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Xsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Xsequential_2/quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
l
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_69565

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_2_layer_call_fn_69707

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_684142
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?#
?
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_69646

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?
J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:
K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?
*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_69108
input_3
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_684052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
0__inference_quantize_layer_1_layer_call_fn_69369

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_684432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_69570

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?O
?
__inference__traced_save_69846
file_prefixD
@savev2_quantize_layer_1_quantize_layer_1_min_read_readvariableopD
@savev2_quantize_layer_1_quantize_layer_1_max_read_readvariableop>
:savev2_quantize_layer_1_optimizer_step_read_readvariableop=
9savev2_quant_reshape_2_optimizer_step_read_readvariableop<
8savev2_quant_conv2d_2_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_2_kernel_min_read_readvariableop8
4savev2_quant_conv2d_2_kernel_max_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_min_read_readvariableopA
=savev2_quant_conv2d_2_post_activation_max_read_readvariableopC
?savev2_quant_max_pooling2d_2_optimizer_step_read_readvariableop=
9savev2_quant_flatten_2_optimizer_step_read_readvariableop;
7savev2_quant_dense_2_optimizer_step_read_readvariableop7
3savev2_quant_dense_2_kernel_min_read_readvariableop7
3savev2_quant_dense_2_kernel_max_read_readvariableop@
<savev2_quant_dense_2_post_activation_min_read_readvariableop@
<savev2_quant_dense_2_post_activation_max_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&BDlayer_with_weights-0/quantize_layer_1_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-0/quantize_layer_1_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_quantize_layer_1_quantize_layer_1_min_read_readvariableop@savev2_quantize_layer_1_quantize_layer_1_max_read_readvariableop:savev2_quantize_layer_1_optimizer_step_read_readvariableop9savev2_quant_reshape_2_optimizer_step_read_readvariableop8savev2_quant_conv2d_2_optimizer_step_read_readvariableop4savev2_quant_conv2d_2_kernel_min_read_readvariableop4savev2_quant_conv2d_2_kernel_max_read_readvariableop=savev2_quant_conv2d_2_post_activation_min_read_readvariableop=savev2_quant_conv2d_2_post_activation_max_read_readvariableop?savev2_quant_max_pooling2d_2_optimizer_step_read_readvariableop9savev2_quant_flatten_2_optimizer_step_read_readvariableop7savev2_quant_dense_2_optimizer_step_read_readvariableop3savev2_quant_dense_2_kernel_min_read_readvariableop3savev2_quant_dense_2_kernel_max_read_readvariableop<savev2_quant_dense_2_post_activation_min_read_readvariableop<savev2_quant_dense_2_post_activation_max_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::: : : : : : : : : : : : : : :::
:	?
: : : : :::
:	?
:::
:	?
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:
:%!

_output_shapes
:	?
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
:
:%!!

_output_shapes
:	?
: "

_output_shapes
::,#(
&
_output_shapes
:: $

_output_shapes
:
:%%!

_output_shapes
:	?
:&

_output_shapes
: 
?
K
/__inference_quant_flatten_2_layer_call_fn_69580

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_686862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_69174

inputs
unknown: 
	unknown_0: #
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?

	unknown_8: 
	unknown_9: 

unknown_10:


unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_689272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69029
input_3 
quantize_layer_1_68994:  
quantize_layer_1_68996: .
quant_conv2d_2_69000:"
quant_conv2d_2_69002:"
quant_conv2d_2_69004:"
quant_conv2d_2_69006:
quant_conv2d_2_69008: 
quant_conv2d_2_69010: &
quant_dense_2_69015:	?

quant_dense_2_69017: 
quant_dense_2_69019: !
quant_dense_2_69021:

quant_dense_2_69023: 
quant_dense_2_69025: 
identity??&quant_conv2d_2/StatefulPartitionedCall?%quant_dense_2/StatefulPartitionedCall?(quantize_layer_1/StatefulPartitionedCall?
(quantize_layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_3quantize_layer_1_68994quantize_layer_1_68996*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_684432*
(quantize_layer_1/StatefulPartitionedCall?
quant_reshape_2/PartitionedCallPartitionedCall1quantize_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_684632!
quant_reshape_2/PartitionedCall?
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(quant_reshape_2/PartitionedCall:output:0quant_conv2d_2_69000quant_conv2d_2_69002quant_conv2d_2_69004quant_conv2d_2_69006quant_conv2d_2_69008quant_conv2d_2_69010*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_684862(
&quant_conv2d_2/StatefulPartitionedCall?
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_685052'
%quant_max_pooling2d_2/PartitionedCall?
quant_flatten_2/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_685132!
quant_flatten_2/PartitionedCall?
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall(quant_flatten_2/PartitionedCall:output:0quant_dense_2_69015quant_dense_2_69017quant_dense_2_69019quant_dense_2_69021quant_dense_2_69023quant_dense_2_69025*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_685352'
%quant_dense_2/StatefulPartitionedCall?
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^quant_conv2d_2/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall)^quantize_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2T
(quantize_layer_1/StatefulPartitionedCall(quantize_layer_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_3
?
f
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_68463

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_68505

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_69586

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_68927

inputs 
quantize_layer_1_68892:  
quantize_layer_1_68894: .
quant_conv2d_2_68898:"
quant_conv2d_2_68900:"
quant_conv2d_2_68902:"
quant_conv2d_2_68904:
quant_conv2d_2_68906: 
quant_conv2d_2_68908: &
quant_dense_2_68913:	?

quant_dense_2_68915: 
quant_dense_2_68917: !
quant_dense_2_68919:

quant_dense_2_68921: 
quant_dense_2_68923: 
identity??&quant_conv2d_2/StatefulPartitionedCall?%quant_dense_2/StatefulPartitionedCall?(quantize_layer_1/StatefulPartitionedCall?
(quantize_layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_1_68892quantize_layer_1_68894*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_688472*
(quantize_layer_1/StatefulPartitionedCall?
quant_reshape_2/PartitionedCallPartitionedCall1quantize_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_688112!
quant_reshape_2/PartitionedCall?
&quant_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(quant_reshape_2/PartitionedCall:output:0quant_conv2d_2_68898quant_conv2d_2_68900quant_conv2d_2_68902quant_conv2d_2_68904quant_conv2d_2_68906quant_conv2d_2_68908*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_687742(
&quant_conv2d_2/StatefulPartitionedCall?
%quant_max_pooling2d_2/PartitionedCallPartitionedCall/quant_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_687022'
%quant_max_pooling2d_2/PartitionedCall?
quant_flatten_2/PartitionedCallPartitionedCall.quant_max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_686862!
quant_flatten_2/PartitionedCall?
%quant_dense_2/StatefulPartitionedCallStatefulPartitionedCall(quant_flatten_2/PartitionedCall:output:0quant_dense_2_68913quant_dense_2_68915quant_dense_2_68917quant_dense_2_68919quant_dense_2_68921quant_dense_2_68923*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_686572'
%quant_dense_2/StatefulPartitionedCall?
IdentityIdentity.quant_dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp'^quant_conv2d_2/StatefulPartitionedCall&^quant_dense_2/StatefulPartitionedCall)^quantize_layer_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2P
&quant_conv2d_2/StatefulPartitionedCall&quant_conv2d_2/StatefulPartitionedCall2N
%quant_dense_2/StatefulPartitionedCall%quant_dense_2/StatefulPartitionedCall2T
(quantize_layer_1/StatefulPartitionedCall(quantize_layer_1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
-__inference_quant_dense_2_layer_call_fn_69609

inputs
unknown:	?

	unknown_0: 
	unknown_1: 
	unknown_2:

	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_685352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?b
?
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_69702

inputs>
+lastvaluequant_rank_readvariableop_resource:	?
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:
@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02$
"LastValueQuant/Rank/ReadVariableOpl
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rankz
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range/startz
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range/delta?
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:2
LastValueQuant/range?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMin?
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02&
$LastValueQuant/Rank_1/ReadVariableOpp
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rank_1~
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range_1/start~
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range_1/delta?
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:2
LastValueQuant/range_1?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?w
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69229

inputs\
Rquantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: ^
Tquantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: q
Wquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:g
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:<
.quant_conv2d_2_biasadd_readvariableop_resource:Z
Pquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: _
Lquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?
X
Nquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: X
Nquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: ;
-quant_dense_2_biasadd_readvariableop_resource:
Y
Oquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: [
Qquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??%quant_conv2d_2/BiasAdd/ReadVariableOp?Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?$quant_dense_2/BiasAdd/ReadVariableOp?Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpRquantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02K
Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpTquantize_layer_1_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02M
Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
:quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsQquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Squantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2<
:quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars?
quant_reshape_2/ShapeShapeDquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*
_output_shapes
:2
quant_reshape_2/Shape?
#quant_reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#quant_reshape_2/strided_slice/stack?
%quant_reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%quant_reshape_2/strided_slice/stack_1?
%quant_reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%quant_reshape_2/strided_slice/stack_2?
quant_reshape_2/strided_sliceStridedSlicequant_reshape_2/Shape:output:0,quant_reshape_2/strided_slice/stack:output:0.quant_reshape_2/strided_slice/stack_1:output:0.quant_reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quant_reshape_2/strided_slice?
quant_reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
quant_reshape_2/Reshape/shape/1?
quant_reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
quant_reshape_2/Reshape/shape/2?
quant_reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
quant_reshape_2/Reshape/shape/3?
quant_reshape_2/Reshape/shapePack&quant_reshape_2/strided_slice:output:0(quant_reshape_2/Reshape/shape/1:output:0(quant_reshape_2/Reshape/shape/2:output:0(quant_reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
quant_reshape_2/Reshape/shape?
quant_reshape_2/ReshapeReshapeDquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0&quant_reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
quant_reshape_2/Reshape?
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02P
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02R
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(2A
?quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
quant_conv2d_2/Conv2DConv2D quant_reshape_2/Reshape:output:0Iquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
quant_conv2d_2/Conv2D?
%quant_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.quant_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%quant_conv2d_2/BiasAdd/ReadVariableOp?
quant_conv2d_2/BiasAddBiasAddquant_conv2d_2/Conv2D:output:0-quant_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
quant_conv2d_2/BiasAdd?
quant_conv2d_2/ReluReluquant_conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
quant_conv2d_2/Relu?
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquant_conv2d_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars!quant_conv2d_2/Relu:activations:0Oquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2:
8quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars?
quant_max_pooling2d_2/MaxPoolMaxPoolBquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
quant_max_pooling2d_2/MaxPool
quant_flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
quant_flatten_2/Const?
quant_flatten_2/ReshapeReshape&quant_max_pooling2d_2/MaxPool:output:0quant_flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
quant_flatten_2/Reshape?
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpLquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?
*
dtype02E
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpNquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02G
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpNquant_dense_2_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype02G
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
4quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsKquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Mquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(26
4quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars?
quant_dense_2/MatMulMatMul quant_flatten_2/Reshape:output:0>quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2
quant_dense_2/MatMul?
$quant_dense_2/BiasAdd/ReadVariableOpReadVariableOp-quant_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02&
$quant_dense_2/BiasAdd/ReadVariableOp?
quant_dense_2/BiasAddBiasAddquant_dense_2/MatMul:product:0,quant_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
quant_dense_2/BiasAdd?
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02H
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_dense_2_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense_2/BiasAdd:output:0Nquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
29
7quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentityAquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp&^quant_conv2d_2/BiasAdd/ReadVariableOpO^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpQ^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Q^quant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2H^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1%^quant_dense_2/BiasAdd/ReadVariableOpD^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpF^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1F^quant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2G^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpI^quant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1J^quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpL^quantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : 2N
%quant_conv2d_2/BiasAdd/ReadVariableOp%quant_conv2d_2/BiasAdd/ReadVariableOp2?
Nquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpNquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Pquant_conv2d_2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22?
Gquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquant_conv2d_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12L
$quant_dense_2/BiasAdd/ReadVariableOp$quant_dense_2/BiasAdd/ReadVariableOp2?
Cquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpCquant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Equant_dense_2/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22?
Fquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpFquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Hquant_dense_2/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Iquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpIquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Kquantize_layer_1/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?b
?
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_68657

inputs>
+lastvaluequant_rank_readvariableop_resource:	?
/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:
@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02$
"LastValueQuant/Rank/ReadVariableOpl
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rankz
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range/startz
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range/delta?
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:2
LastValueQuant/range?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMin?
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02&
$LastValueQuant/Rank_1/ReadVariableOpp
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rank_1~
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range_1/start~
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range_1/delta?
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:2
LastValueQuant/range_1?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?
*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?
*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????
2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?'
?
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_68486

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_quant_reshape_2_layer_call_fn_69418

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_688112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_69967
file_prefix@
6assignvariableop_quantize_layer_1_quantize_layer_1_min: B
8assignvariableop_1_quantize_layer_1_quantize_layer_1_max: <
2assignvariableop_2_quantize_layer_1_optimizer_step: ;
1assignvariableop_3_quant_reshape_2_optimizer_step: :
0assignvariableop_4_quant_conv2d_2_optimizer_step: :
,assignvariableop_5_quant_conv2d_2_kernel_min::
,assignvariableop_6_quant_conv2d_2_kernel_max:?
5assignvariableop_7_quant_conv2d_2_post_activation_min: ?
5assignvariableop_8_quant_conv2d_2_post_activation_max: A
7assignvariableop_9_quant_max_pooling2d_2_optimizer_step: <
2assignvariableop_10_quant_flatten_2_optimizer_step: :
0assignvariableop_11_quant_dense_2_optimizer_step: 6
,assignvariableop_12_quant_dense_2_kernel_min: 6
,assignvariableop_13_quant_dense_2_kernel_max: ?
5assignvariableop_14_quant_dense_2_post_activation_min: ?
5assignvariableop_15_quant_dense_2_post_activation_max: '
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: /
!assignvariableop_21_conv2d_2_bias:=
#assignvariableop_22_conv2d_2_kernel:.
 assignvariableop_23_dense_2_bias:
5
"assignvariableop_24_dense_2_kernel:	?
#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: 6
(assignvariableop_29_adam_conv2d_2_bias_m:D
*assignvariableop_30_adam_conv2d_2_kernel_m:5
'assignvariableop_31_adam_dense_2_bias_m:
<
)assignvariableop_32_adam_dense_2_kernel_m:	?
6
(assignvariableop_33_adam_conv2d_2_bias_v:D
*assignvariableop_34_adam_conv2d_2_kernel_v:5
'assignvariableop_35_adam_dense_2_bias_v:
<
)assignvariableop_36_adam_dense_2_kernel_v:	?

identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&BDlayer_with_weights-0/quantize_layer_1_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-0/quantize_layer_1_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-2/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp6assignvariableop_quantize_layer_1_quantize_layer_1_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp8assignvariableop_1_quantize_layer_1_quantize_layer_1_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp2assignvariableop_2_quantize_layer_1_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp1assignvariableop_3_quant_reshape_2_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp0assignvariableop_4_quant_conv2d_2_optimizer_stepIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_quant_conv2d_2_kernel_minIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_quant_conv2d_2_kernel_maxIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp5assignvariableop_7_quant_conv2d_2_post_activation_minIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_quant_conv2d_2_post_activation_maxIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_quant_max_pooling2d_2_optimizer_stepIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp2assignvariableop_10_quant_flatten_2_optimizer_stepIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_quant_dense_2_optimizer_stepIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_quant_dense_2_kernel_minIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_quant_dense_2_kernel_maxIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp5assignvariableop_14_quant_dense_2_post_activation_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_quant_dense_2_post_activation_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_conv2d_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_2_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_conv2d_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_2_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_2_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_2_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_2_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_2_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37f
Identity_38IdentityIdentity_37:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_38?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
-__inference_quant_dense_2_layer_call_fn_69626

inputs
unknown:	?

	unknown_0: 
	unknown_1: 
	unknown_2:

	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_686572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_68443

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_68686

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_69408

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity??#AllValuesQuantize/AssignMaxAllValue?#AllValuesQuantize/AssignMinAllValue?8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?(AllValuesQuantize/Maximum/ReadVariableOp?(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2
AllValuesQuantize/Const?
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin?
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
AllValuesQuantize/Const_1?
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMax?
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum?
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y?
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1?
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOp?
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum?
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y?
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1?
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue?
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*+
_output_shapes
:?????????2+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_quant_reshape_2_layer_call_fn_69413

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_684632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_69432

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_34
serving_default_input_3:0?????????A
quant_dense_20
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
quantize_layer_1_min
quantize_layer_1_max
quantizer_vars
optimizer_step
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	layer
optimizer_step
_weight_vars
_quantize_activations
_output_quantizers
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	layer
optimizer_step
 _weight_vars
!
kernel_min
"
kernel_max
#_quantize_activations
$post_activation_min
%post_activation_max
&_output_quantizers
'regularization_losses
(trainable_variables
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	+layer
,optimizer_step
-_weight_vars
._quantize_activations
/_output_quantizers
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	4layer
5optimizer_step
6_weight_vars
7_quantize_activations
8_output_quantizers
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	=layer
>optimizer_step
?_weight_vars
@
kernel_min
A
kernel_max
B_quantize_activations
Cpost_activation_min
Dpost_activation_max
E_output_quantizers
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Jiter

Kbeta_1

Lbeta_2
	Mdecay
Nlearning_rateOm?Pm?Qm?Rm?Ov?Pv?Qv?Rv?"
	optimizer
 "
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
?
0
1
2
3
O4
P5
6
!7
"8
$9
%10
,11
512
Q13
R14
>15
@16
A17
C18
D19"
trackable_list_wrapper
?
regularization_losses
Snon_trainable_variables
	trainable_variables

	variables
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics

Wlayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+ 2%quantize_layer_1/quantize_layer_1_min
-:+ 2%quantize_layer_1/quantize_layer_1_max
:
min_var
max_var"
trackable_dict_wrapper
':% 2quantize_layer_1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
regularization_losses
Xnon_trainable_variables
trainable_variables
	variables
Ymetrics
Zlayer_regularization_losses
[layer_metrics

\layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
&:$ 2quant_reshape_2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
anon_trainable_variables
trainable_variables
	variables
bmetrics
clayer_regularization_losses
dlayer_metrics

elayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Pkernel
Obias
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
%:# 2quant_conv2d_2/optimizer_step
'
j0"
trackable_list_wrapper
%:#2quant_conv2d_2/kernel_min
%:#2quant_conv2d_2/kernel_max
 "
trackable_list_wrapper
*:( 2"quant_conv2d_2/post_activation_min
*:( 2"quant_conv2d_2/post_activation_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
Q
O0
P1
2
!3
"4
$5
%6"
trackable_list_wrapper
?
'regularization_losses
knon_trainable_variables
(trainable_variables
)	variables
lmetrics
mlayer_regularization_losses
nlayer_metrics

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
pregularization_losses
qtrainable_variables
r	variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
,:* 2$quant_max_pooling2d_2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
?
0regularization_losses
tnon_trainable_variables
1trainable_variables
2	variables
umetrics
vlayer_regularization_losses
wlayer_metrics

xlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
&:$ 2quant_flatten_2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
?
9regularization_losses
}non_trainable_variables
:trainable_variables
;	variables
~metrics
layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Rkernel
Qbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
$:" 2quant_dense_2/optimizer_step
(
?0"
trackable_list_wrapper
 : 2quant_dense_2/kernel_min
 : 2quant_dense_2/kernel_max
 "
trackable_list_wrapper
):' 2!quant_dense_2/post_activation_min
):' 2!quant_dense_2/post_activation_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
Q
Q0
R1
>2
@3
A4
C5
D6"
trackable_list_wrapper
?
Fregularization_losses
?non_trainable_variables
Gtrainable_variables
H	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:2conv2d_2/bias
):'2conv2d_2/kernel
:
2dense_2/bias
!:	?
2dense_2/kernel
?
0
1
2
3
4
!5
"6
$7
%8
,9
510
>11
@12
A13
C14
D15"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]regularization_losses
?non_trainable_variables
^trainable_variables
_	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
?
fregularization_losses
?non_trainable_variables
gtrainable_variables
h	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
P0
?2"
trackable_tuple_wrapper
C
0
!1
"2
$3
%4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pregularization_losses
?non_trainable_variables
qtrainable_variables
r	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
yregularization_losses
?non_trainable_variables
ztrainable_variables
{	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?	variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
R0
?2"
trackable_tuple_wrapper
C
>0
@1
A2
C3
D4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
=0"
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:
!min_var
"max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:
@min_var
Amax_var"
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 :2Adam/conv2d_2/bias/m
.:,2Adam/conv2d_2/kernel/m
:
2Adam/dense_2/bias/m
&:$	?
2Adam/dense_2/kernel/m
 :2Adam/conv2d_2/bias/v
.:,2Adam/conv2d_2/kernel/v
:
2Adam/dense_2/bias/v
&:$	?
2Adam/dense_2/kernel/v
?B?
 __inference__wrapped_model_68405input_3"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sequential_2_layer_call_fn_68581
,__inference_sequential_2_layer_call_fn_69141
,__inference_sequential_2_layer_call_fn_69174
,__inference_sequential_2_layer_call_fn_68991?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69229
G__inference_sequential_2_layer_call_and_return_conditional_losses_69360
G__inference_sequential_2_layer_call_and_return_conditional_losses_69029
G__inference_sequential_2_layer_call_and_return_conditional_losses_69067?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_quantize_layer_1_layer_call_fn_69369
0__inference_quantize_layer_1_layer_call_fn_69378?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_69387
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_69408?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_quant_reshape_2_layer_call_fn_69413
/__inference_quant_reshape_2_layer_call_fn_69418?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_69432
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_69446?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_quant_conv2d_2_layer_call_fn_69463
.__inference_quant_conv2d_2_layer_call_fn_69480?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_69501
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_69550?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_quant_max_pooling2d_2_layer_call_fn_69555
5__inference_quant_max_pooling2d_2_layer_call_fn_69560?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_69565
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_69570?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_quant_flatten_2_layer_call_fn_69575
/__inference_quant_flatten_2_layer_call_fn_69580?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_69586
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_69592?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_quant_dense_2_layer_call_fn_69609
-__inference_quant_dense_2_layer_call_fn_69626?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_69646
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_69702?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkwjkwargs
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_69108input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_2_layer_call_fn_69707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_69712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_68405?P!"O$%R@AQCD4?1
*?'
%?"
input_3?????????
? "=?:
8
quant_dense_2'?$
quant_dense_2?????????
?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_69712?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_69707?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_69501tP!"O$%;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
I__inference_quant_conv2d_2_layer_call_and_return_conditional_losses_69550tP!"O$%;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
.__inference_quant_conv2d_2_layer_call_fn_69463gP!"O$%;?8
1?.
(?%
inputs?????????
p 
? " ???????????
.__inference_quant_conv2d_2_layer_call_fn_69480gP!"O$%;?8
1?.
(?%
inputs?????????
p
? " ???????????
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_69646eR@AQCD4?1
*?'
!?
inputs??????????
p 
? "%?"
?
0?????????

? ?
H__inference_quant_dense_2_layer_call_and_return_conditional_losses_69702eR@AQCD4?1
*?'
!?
inputs??????????
p
? "%?"
?
0?????????

? ?
-__inference_quant_dense_2_layer_call_fn_69609XR@AQCD4?1
*?'
!?
inputs??????????
p 
? "??????????
?
-__inference_quant_dense_2_layer_call_fn_69626XR@AQCD4?1
*?'
!?
inputs??????????
p
? "??????????
?
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_69586e;?8
1?.
(?%
inputs?????????
p 
? "&?#
?
0??????????
? ?
J__inference_quant_flatten_2_layer_call_and_return_conditional_losses_69592e;?8
1?.
(?%
inputs?????????
p
? "&?#
?
0??????????
? ?
/__inference_quant_flatten_2_layer_call_fn_69575X;?8
1?.
(?%
inputs?????????
p 
? "????????????
/__inference_quant_flatten_2_layer_call_fn_69580X;?8
1?.
(?%
inputs?????????
p
? "????????????
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_69565l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
P__inference_quant_max_pooling2d_2_layer_call_and_return_conditional_losses_69570l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
5__inference_quant_max_pooling2d_2_layer_call_fn_69555_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
5__inference_quant_max_pooling2d_2_layer_call_fn_69560_;?8
1?.
(?%
inputs?????????
p
? " ???????????
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_69432h7?4
-?*
$?!
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
J__inference_quant_reshape_2_layer_call_and_return_conditional_losses_69446h7?4
-?*
$?!
inputs?????????
p
? "-?*
#? 
0?????????
? ?
/__inference_quant_reshape_2_layer_call_fn_69413[7?4
-?*
$?!
inputs?????????
p 
? " ???????????
/__inference_quant_reshape_2_layer_call_fn_69418[7?4
-?*
$?!
inputs?????????
p
? " ???????????
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_69387h7?4
-?*
$?!
inputs?????????
p 
? ")?&
?
0?????????
? ?
K__inference_quantize_layer_1_layer_call_and_return_conditional_losses_69408h7?4
-?*
$?!
inputs?????????
p
? ")?&
?
0?????????
? ?
0__inference_quantize_layer_1_layer_call_fn_69369[7?4
-?*
$?!
inputs?????????
p 
? "???????????
0__inference_quantize_layer_1_layer_call_fn_69378[7?4
-?*
$?!
inputs?????????
p
? "???????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_69029uP!"O$%R@AQCD<?9
2?/
%?"
input_3?????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69067uP!"O$%R@AQCD<?9
2?/
%?"
input_3?????????
p

 
? "%?"
?
0?????????

? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69229tP!"O$%R@AQCD;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_69360tP!"O$%R@AQCD;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????

? ?
,__inference_sequential_2_layer_call_fn_68581hP!"O$%R@AQCD<?9
2?/
%?"
input_3?????????
p 

 
? "??????????
?
,__inference_sequential_2_layer_call_fn_68991hP!"O$%R@AQCD<?9
2?/
%?"
input_3?????????
p

 
? "??????????
?
,__inference_sequential_2_layer_call_fn_69141gP!"O$%R@AQCD;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
,__inference_sequential_2_layer_call_fn_69174gP!"O$%R@AQCD;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
#__inference_signature_wrapper_69108?P!"O$%R@AQCD??<
? 
5?2
0
input_3%?"
input_3?????????"=?:
8
quant_dense_2'?$
quant_dense_2?????????
