import subprocess

def exec (shell_command: str) -> str:
    '''
    Executes a shell command and returns the catched stdout.
    
    :param shell_command: shell command to execute
    :type shell_command: str
    :return: the stdout of the process
    :rtype: str
    '''
    result = subprocess.run(
        shell_command, 
        capture_output=True,  # Capture stdout and stderr
        text=True,  # Decode stdout and stderr as text using default encoding
        check=True,
        shell=True)
    if result.stderr:
        return result.stdout + '\n' + result.stderr
    return result.stdout

class hipBlasLt:

    '''
    Main hipBlasLt object.

    gpuName     (Optional) Allows to provide a string for GPU name

    GEMM context:
        D = Activation( α * op(A) * op(B) + β * op(C) + bias )

        - A: Input matrix (e.g., activations or previous layer output)
        - B: Weight matrix (e.g., layer weights)
        - C: Optional matrix added to the result (same shape as D), often used for residuals or drift correction
        - D: Output matrix after GEMM and optional activation
        - α (alpha): Scalar multiplier for the product op(A)*op(B)
        - β (beta): Scalar multiplier for op(C)
    '''

    def __init__(self, gpuName: str=''):
        # globals
        self.gpuName = gpuName
        self.command = ''
        self.supportedPrecisions = [8, 16, 32]
        self.alpha = 1.0
        self.beta = 1.0
        self.matrixDimensions = {
            'A': (0,0),
            'B': (0,0)
        }
        self.matrixPrecision = {
            'A': 'f32_r',
            'B': 'f32_r',
            'C': 'f32_r',
            'D': 'f32_r'
        }
        self.transpose = {
            'A': False,
            'B': False
        }
        self.biasPrecision = ''
        self.biasSource = ''
        self.results = ''
        self.supportedActivations = ['none', 'gelu', 'relu', 'swish', 'clamp']
        self.activation = ''
        self.computePrecision = ''
        self.batchSize = 1
    
    def specifyMatrix (self, matrixName: str, rows: int, cols: int, precision: int, transpose: bool=False, bfloat: bool=False) -> None:
        
        ''' 
        Sets the matrix dimension for provided matrix A, B, C.

        matrixName              Matrix variable name string 'a' or 'A', not case sensitive.
        '''

        # Convert to upper casing
        matrixName = matrixName.upper()

        # Format the precision parameter, and add bfloat type if selected
        if precision not in self.supportedPrecisions:
            raise ValueError(f'Precision "{precision}" is not supported, please choose a precision from {self.supportedPrecisions}.')
        precision = self.convertPrecision(precision)
        self.matrixPrecision[matrixName] = precision

        # Continue only for A and B matrices as all others will automatically build
        if matrixName not in 'AB':
            return

        # Sanity checks
        if matrixName not in self.matrixDimensions.keys():
            raise KeyError(f'No matrix named "{matrixName}", please use the letters A, B')
        if matrixName == 'B' and self.matrixDimensions['A'][1] != 0 and rows != self.matrixDimensions['A'][1]:
            raise ValueError(f'The row dimension of matrix B (k, n) must equal the col dimensions of matrix A (m, k)!')
        elif matrixName == 'A' and self.matrixDimensions['B'][0] != 0 and cols != self.matrixDimensions['B'][0]:
            raise ValueError(f'The col dimension of matrix A (m, k) must equal the row dimensions of matrix B (k, n)!')

        self.matrixDimensions[matrixName] = (rows, cols)
        self.transpose[matrixName] = transpose
    
    def addBias (self, precision: int, bfloat: bool=False, source: str='d'):

        ''' 
        Applies a bias to the final GEMM with provided precision.
        '''

        self.biasPrecision = self.convertPrecision(precision, bfloat)
        self.biasSource = source

    def setActivation (self, activationFunction: str) -> None:

        '''
        Set the activation function type.
        
        :param activationFunction: The activation function none, gelu, relu, swish, clamp
        :type activationFunction: str
        '''
        
        activationFunction = activationFunction.lower()
        if not activationFunction in self.supportedActivations:
            raise ValueError(f'The provided activation function "{activationFunction}" is not supported. Please choose an activation from the supported list: none, gelu, relu, swish, clamp')
        self.activation = activationFunction # save globally

    def specifyScalars (self, alpha: float, beta: float) -> None:

        ''' 
        Denotes the provided alpha, beta and gamma factors and makes them accessible globally.
        
        alpha           scalar factor in GEMM multiplication alpha * A x B
        beta            scalar bias factor
        gamma           
        '''

        self.alpha = alpha
        self.beta  = beta

    def setComputePrecision (self, precision: str|int) -> None:

        '''
        Sets the general compute precision - important in mixed precision.
        
        :param self: Description
        :param precision: The current precision string s,f32_r,x,xf32_r,f64_r,i32_r,f32_bf16_r or integer 32, 64.
        :type precision: str
        '''

        if isinstance(precision, int):
            self.computePrecision = self.convertPrecision(precision)
        else:
            self.computePrecision = precision

    def convertPrecision (self, precision: int, bfloat: bool=False) -> str:
        '''
        Converts integer precisions in bit representation e.g. 32bit to a hipblaslt-compliant string format. 
        
        :param precision: Description
        :type precision: int
        :param bfloat: If to apply bfloat
        :return: final precision string format
        :rtype: str
        '''
        string = f'f{precision}_r' if precision > 8 else 'i8_r'
        if precision == 16 and bfloat:
            string = 'b' + string
        return string
    
    def run (self, batchSize: int=1, validate: bool=False, warmupIterations: int=0) -> None:

        ''' 
        Runs a benchmark based on stored parameters.

        batchSize           Total number of GEMMs to compute in parallel
        validate            Performs GPU validation using the CPU
        '''

        # ---- Construct the command ----
        self.command = './hipblaslt-bench --function matmul '
        # Add the dimensions
        m = self.matrixDimensions["A"][0]
        n = self.matrixDimensions["B"][1]
        k = self.matrixDimensions["A"][1]
        self.command += f'-m {m} -n {n} -k {k} '
        # Transpose
        self.command += f'--transA {"T" if self.transpose["A"] else "N"} '
        self.command += f'--transB {"T" if self.transpose["B"] else "N"} ' 
        # Add the spride sizes for faster sourcing
        self.command += f'--lda {m} --ldb {k} --ldc {m} --ldd {m} '
        # Add the scalars
        self.command += f'--alpha {self.alpha} --beta {self.beta} '
        # Add batch size
        if batchSize > 1:
            self.command += f'--batch_count {batchSize} '
        # Add a bias if enabled
        if self.biasPrecision:
            self.command += f'--bias_vector --bias_type {self.biasPrecision} --bias_source {self.biasSource} '
        # Add the precisions
        if self.computePrecision:
            self.command += f'--compute_type {self.computePrecision} '
        # self.command = f'--a_type {self.matrixPrecision["A"]} --b_type {self.matrixPrecision["B"]} --c_type {self.matrixPrecision["C"]}'
        self.command += '--a_type {} --b_type {} --c_type {} --d_type {} '.format(*self.matrixPrecision.values())
        # Warmup iterations before timing starts
        if warmupIterations:
            self.command += f'-j {warmupIterations} '
        # Add cpu validation
        if validate:
            self.command += f'-v'

        # ---- Run Benchmark ----
        try:
            print("[hipBLASlt-api] Start benchmark ...")
            print(f"[hipBLASlt-api] Will run start command:\n{self.command}")
            # Start benchmark process and collect results
            self.results = exec(self.command)
            self.results = '\n'.join(self.results.split('\n')[-3:-1])
        except Exception as e:
            print("[hipBLASlt-api] Error:", e)
            return
        finally:
            print("[hipBLASlt-api] Finished benchmark ...")

        self.batchSize = batchSize
    
    def showResults (self, *metrics: str) -> None:

        '''
        Print results in the terminal
        
        :param self: Description
        :param metrics: Description
        :type metrics: str
        '''

        # Remove whitespaces
        results = ''.join(self.results.split(' '))
        # Split data in keys and values
        keys, values = results.split('\n')
        keys = keys.split(':')[1]
        keys = keys.split(',')
        values = values.split(',')
        cellSize = 20
        # Aggregate more info
        info =  f'GPU                       : {self.gpuName}\n' if self.gpuName else ''
        info += f'Matrix Dimensions (m,n,k) : {self.matrixDimensions["A"][0]}, {self.matrixDimensions["B"][1]}, {self.matrixDimensions["A"][1]}\n'
        info += f'Precisions (A,B,C,D,bias) : {self.matrixPrecision["A"]}, {self.matrixPrecision["B"]}, {self.matrixPrecision["C"]}, {self.matrixPrecision["D"]}, {self.biasPrecision}\n' 
        info += f'Batch Size                : {self.batchSize}\n'
        info += f'Alpha                     : {self.batchSize}\n'
        info += f'Beta                      : {self.batchSize}\n'
        info += f'Bias Vector               : {"Yes" if self.biasPrecision else "No"}\n'
        info += f'Transpose (A, B)          : {list(self.transpose.values())}\n\n'
        # Create header
        header = ''
        row = ''
        for i in range(len(keys)):
            key = keys[i]
            if key in metrics:
                padLength = cellSize - len(key) if len(key) <= cellSize else 0
                header += key + ' ' * padLength
                val = values[i]
                padLength = cellSize - len(val) if len(val) <= cellSize else 0
                row += values[i] + ' ' * padLength

        print('-'*8 + 'Benchmark results' + '-'*8 + '\n' + info + header + '\n' + row)
    
    def getResultData (self, *metrics: str) -> dict[str, str|float|int]:

        # Remove whitespaces
        results = ''.join(self.results.split(' '))
        # Split data in keys and values
        keys, values = results.split('\n')
        keys = keys.split(':')[1]
        keys = keys.split(',')
        values = values.split(',')
        data = {}

        # Aggregate metrics
        for i in range(len(keys)):
            key = keys[i]
            if key in metrics:
                val = values[i]
                try:
                    data[key] = float(val)
                except:
                    data[key] = val

        return data

def simpleGemmBenchmark (gpuName: str='', activation: str='relu'):

    '''
    A simple GEMM benchmark.
    
    :param gpuName: Description
    :type gpuName: str
    :param activation: Description
    :type activation: str
    '''


    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    precisions = [16, 32]
    metrics = ['batch_count', 'hipblaslt-Gflops', 'hipblaslt-GB/s', 'us', 'CPU-Gflops', 'CPU-us']

    data = []
    for precision in precisions:
        for dim in sizes:
            # Perform benchmark
            # Initialize a fresh hipBlasLt object
            bench = hipBlasLt(gpuName)
            bench.specifyMatrix('A', dim, dim, precision)
            bench.specifyMatrix('B', dim, dim, precision)
            bench.specifyMatrix('C', dim, dim, precision)
            bench.specifyMatrix('D', dim, dim, precision)
            bench.setComputePrecision(32)
            bench.specifyScalars(1.0, 1.0)
            bench.setActivation(activation)
            bench.run(1, True)
            # Denote data row
            dataRow = [precision, dim, activation] + list(bench.getResultData(*metrics).values())
            data.append(dataRow)

    # Create table
    header = ''
    row = ''
    cellSize = 20
    # Create table header
    keys = ['precision', 'tensor_size', 'activation'] + metrics
    for i in range(len(keys)): # header
        key = keys[i]
        padLength = cellSize - len(key) if len(key) <= cellSize else 0
        header += key + ' ' * padLength
    print(header)
    # Create table rows
    for row in data: # data rows
        line = ''
        for i in range(len(keys)):
            val = row[i]
            padLength = cellSize - len(str(val)) if len(str(val)) <= cellSize else 0
            line += str(val) + ' ' * padLength
        print(line)

if __name__ == '__main__':

    # example usage
    hipblt = hipBlasLt('AMD Instinct MI300X')
    hipblt.specifyMatrix('A', 128, 128, 32)
    hipblt.specifyMatrix('B', 128, 128, 32)
    hipblt.setComputePrecision(32)
    hipblt.specifyScalars(1.0, 1.0)
    hipblt.run(batchSize=1, validate=True)
    # show results of the benchmark
    hipblt.showResults('batchcount', 'hipblaslt-Gflops', 'hipblaslt-GB/s', 'us', 'CPU-Gflops', 'CPU-us')
