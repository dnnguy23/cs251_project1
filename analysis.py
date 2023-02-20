'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Duc Nguyen
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from data import Data


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data


    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        return np.min(selected_data, axis = 0)
    
    

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        return np.max(selected_data, axis = 0)
        
        

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        mins = self.min(headers, rows)
        maxes = self.max(headers, rows)
        return mins, maxes
        
        
    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        sums = np.sum(selected_data, axis = 0)
        if (len(rows) != 0):
            return sums/len(rows)
        else:
            return sums/self.data.get_num_samples()
        
                

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        selected_data = self.data.select_data(headers, rows)
        means = self.mean(headers, rows)
        sample_count = len(rows) if len(rows) != 0 else self.data.get_num_samples()
        return np.sum(np.power(selected_data - means, 2), axis = 0) / (sample_count-1)
        
        

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        return np.sqrt(self.var(headers, rows))
        

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()


    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        x = self.data.select_data([ind_var])
        y = self.data.select_data([dep_var])
        plt.scatter(x, y)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)    
        return x, y    
        
        

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        
        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize=fig_sz,
                                sharex="col", sharey="row")
        fig.suptitle(title)
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                # if the same variable, use histogram, else use scatterplot
                if i == j:
                    plt_data = self.data.select_data([data_vars[i]])
                    axes[i, j].hist(plt_data)                    
                else:
                    plt_data = self.data.select_data([data_vars[j], data_vars[i]])
                    axes[i, j].scatter(plt_data[:, 0], plt_data[:, 1])

                # set labels on x-axis and y-axis
                if i == len(data_vars) - 1:
                    axes[i, j].set_xlabel(data_vars[j])
                if j == 0:
                    axes[i, j].set_ylabel(data_vars[i])

        plt.tight_layout()
        return fig, axes
    
    
    '''
    ------------------------------------------
    EXTENSION
    '''
    
    def corr(self, headers):
        '''
        Compute the Pearson correlation coefficients of quantitative variables 
        
        Parameters:
        -----------
        headers: Python list of str.
            Variables whose correlations are pair-wise computed
            
        Returns:
        -----------
        corr_matrix: a Pearson correlation coefficient matrix
        '''
        
        corr_matrix = np.ones((len(headers), len(headers)))
        
        for i in range(len(headers)):
            for j in range(len(headers)):
                if (i == j):
                    continue
                
                x = self.data.select_data([headers[i]])
                y = self.data.select_data([headers[j]])
                
                mean_x = self.mean([headers[i]])
                mean_y = self.mean([headers[j]])
                dev_x = x - mean_x
                dev_y = y - mean_y
        
                num = np.sum(dev_x * dev_y)
                den = np.sqrt(np.sum(dev_x**2) * np.sum(dev_y**2))
                corr_matrix[i,j] = num/den
                
        return corr_matrix
    
    
    def format_corr_matrix (self, headers):
        '''
        returns a string of formatted Pearson correlation matrix among variables
        specfied in headers.
        
        NOTE: We need this function because we cannot name rows and columns of numpy ndarray like 
        pandas dataframe. Lack of variables' name may confuse users.
        '''
        corr_matrix = self.corr(headers)
        
        matrix_str = "\t\t"
        for i in range(len(headers)):
            matrix_str += f"{headers[i]:<10}\t"
        matrix_str += "\n"
        for i in range(len(headers)):
            matrix_str += f"{headers[i]:<10}\t"
            for j in range(len(headers)):
                matrix_str += f"{corr_matrix[i, j]:.8f}\t"
            matrix_str += "\n"
            
        return matrix_str
        
        
        
        
