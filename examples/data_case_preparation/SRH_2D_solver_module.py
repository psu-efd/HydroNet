"""
A module with the function to call SRH-2D. This function has to be in a separate Python file for the parallelization
to work. 

Two functions are defined here:
1. run_one_SRH_2D_case: Run a single SRH-2D case with the specified case_ID. The ManningN value is used to set the Manning's n for the channel.
2. run_one_SRH_2D_case_with_hydrograph: Run a single SRH-2D case with the specified case_ID. The hydrograph file is used to set the inflow boundary condition.
"""

import multiprocessing
import os
import shutil

from pathlib import Path

import pyHMT2D

def run_one_SRH_2D_case(case_ID, srhcontrol_file, ManningN_MaterialID, ManningN, ManningN_MaterialName, Q_bc_ID, Q,  WSE_bc_ID, WSE, system_name, bDeleteCaseDir=True):
    """
    Run a single SRH-2D case with the specified case_ID. The ManningN value is used to set the Manning's n for the channel.
    The Q and WSE values are used to set the boundary conditions.

    Parameters
    ----------
    case_ID : int
        ID of the case to run
    srhcontrol_file : str
        Name of the SRH-2D control file (either srhhydro or SIF file), e.g., "case_SIF.dat" or "case.srhhydro"
    ManningN_MaterialID : int
        ID of the Manning's n zone. Note: In HEC-RAS, the ID is 1-based. We need -1 to make it 0-based in Python.
    ManningN : float
        Manning's n value for this case
    ManningN_MaterialName : str
        Name of the Manning's n zone
    Q_bc_ID : int
        ID of the inlet boundary condition
    Q : float
        Flow rate for the inlet boundary condition
    WSE_bc_ID : int
        ID of the outlet boundary condition
    WSE : float
        Water surface elevation for the outlet boundary condition
    system_name : str
        Name of the system (Windows or Linux)
    bDeleteCaseDir : bool
        whether to delete case directory when simulation is done. Default is True. If it is set to False, the case directory
        will be kept. However, be cautious on this. If the number of simulations is large, all cases together will use
        very large space on the hard disk.

        In this example, the SRH-2D result is saved into a VTK file (thus the case directory is not needed after
        simulation is done).

    Returns
    -------

    """

    processID = multiprocessing.current_process()
    print(processID, ": running SRH-2D case: ", case_ID)

    #base SRH-2D case directory
    base_case_dir = 'base_case'

    #name of directory for the new case, e.g., case_000001, case_000002, etc.
    case_name = 'case_'+str(case_ID).zfill(6)
    new_case_dir = 'cases/'+case_name

    #copy the base SRH-2D case folder
    destination = shutil.copytree(base_case_dir, new_case_dir, dirs_exist_ok=True)

    #chdir to the new case directory
    os.chdir(destination)

    #create a SRH-2D model instance
    if system_name == "Windows":
        #the following should be modified based on your installation of SRH-2D on Windows
        version = "3.6.5"
        srh_pre_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
        srh_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console_v365.exe"
        extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"
    elif system_name == "Linux":
        #the following should be modified based on your installation of SRH-2D on Linux (here we assume the srh2dpre and srh2d executables are in the case directory)
        version = "3.6.2"
        srh_pre_path = r"./srh2dpre"
        srh_path = r"./srh2d"
        extra_dll_path = r"./"
    else:
        raise ValueError("Unsupported operating system: " + system_name)

    #create a SRH-2D model instance
    my_srh_2d_model = pyHMT2D.SRH_2D.SRH_2D_Model(version, srh_pre_path,
                       srh_path, extra_dll_path, faceless=False)

    #initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    #open a SRH-2D project
    my_srh_2d_model.open_project(srhcontrol_file)

    #get the SRH-2D data
    my_srh_2d_data = my_srh_2d_model.get_simulation_case()

    #modify the Manning's n
    if ManningN_MaterialID > 0:
        ManningN_MaterialIDs = [ManningN_MaterialID]    
        ManningNs = [ManningN]
        ManningN_MaterialNames = [ManningN_MaterialName]
        my_srh_2d_data.modify_ManningsNs(ManningN_MaterialIDs, ManningNs, ManningN_MaterialNames)

    #modify the inlet flow rate
    if Q_bc_ID > 0:
        inlet_q_bc_IDs = [Q_bc_ID]
        new_inlet_q_values = [Q]
        my_srh_2d_data.modify_InletQ(inlet_q_bc_IDs, new_inlet_q_values)

    #modify the outlet constant water surface elevation
    if WSE_bc_ID > 0:
        exit_h_bc_IDs = [WSE_bc_ID]
        new_exit_h_values = [WSE]
        my_srh_2d_data.modify_ExitH(exit_h_bc_IDs, new_exit_h_values)

    #save the srhcontrol file after the updates of parameters
    if system_name == "Windows":
        my_srh_2d_data.srhhydro_obj.save_as()  #without any argument, the original filename will be used
    elif system_name == "Linux":
        my_srh_2d_data.srhsif_obj.save_as()    #without any argument, the original filename will be used

    #run SRH-2D Pre to preprocess the case
    bRunSuccessful = my_srh_2d_model.run_pre_model()

    #run the SRH-2D model's current project if SRH-2D Preprocessing is successful
    if bRunSuccessful:
        if system_name == "Windows":
            bRunSuccessful = my_srh_2d_model.run_model()
        elif system_name == "Linux":        #On Linux, the sleepTime and bShowProgress are not supported (there is some issue with the case_INF.DAT file)
            bRunSuccessful = my_srh_2d_model.run_model(sleepTime=10.0, bShowProgress=False)

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()

    #do postprocessing only if bRunSuccessful is true
    if bRunSuccessful:
        #convert SRH-2D result to VTK (This is hard-coded; needs to be changes for a specific case)
        my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data(srhcontrol_file)

        #wether the result is nodal or cell center
        bNodal = False

        if system_name == "Windows":
            #get case's base name 
            case_base_name = my_srh_2d_data.srhhydro_obj.srhhydro_content["Case"]
            if not bNodal:
                xmdf_file_name = case_base_name + "_XMDFC.h5"
            else:
                xmdf_file_name = case_base_name + "_XMDF.h5"

            #read the XMDF file
            my_srh_2d_data.readSRHXMDFFile(xmdf_file_name, bNodal)

            #export the XMDF data to VTK
            vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')
        elif system_name == "Linux":
            #read the SRHC files
            my_srh_2d_data.readSRHCFiles(my_srh_2d_data.srhsif_obj.srhsif_content["Case"])

            #export the SRHC data to VTK
            vtkFileNameList = my_srh_2d_data.outputSRHCDataToVTK(lastTimeStep=True, dir='')

        #copy the vtk result file (only the last time step) to "cases" directory (one level above)
        shutil.copy(vtkFileNameList[-1], "../"+case_name+".vtk")

    # go back to the root
    os.chdir("../..")

    # delete the case folder
    if bDeleteCaseDir:
        shutil.rmtree(destination)

    #if successful, return case_ID; otherwise, return -case_ID
    if bRunSuccessful:
        return  case_ID
    else:
        return -case_ID


def run_one_SRH_2D_case_with_multiple_inlet_q(case_ID, srhcontrol_file, Q_bc_IDs, Qs, system_name, bDeleteCaseDir=True):
    """
    Run a single SRH-2D case with the specified case_ID. The Q values are used to set the boundary conditions for multiple inlets in the domain.

    Parameters
    ----------
    case_ID : int
        ID of the case to run
    srhcontrol_file : str
        Name of the SRH-2D control file (either srhhydro or SIF file), e.g., "case_SIF.dat" or "case.srhhydro"
    Q_bc_IDs : list of int
        ID of the inlet boundary conditions. Note: In HEC-RAS, the ID is 1-based. We need -1 to make it 0-based in Python.
    Qs : list of float
        Flow rate for the inlet boundary conditions
    system_name : str
        Name of the system (Windows or Linux)
    bDeleteCaseDir : bool
        whether to delete case directory when simulation is done. Default is True. If it is set to False, the case directory
        will be kept. However, be cautious on this. If the number of simulations is large, all cases together will use
        very large space on the hard disk.

    Returns
    -------

    """
    
    processID = multiprocessing.current_process()
    print(processID, ": running SRH-2D case: ", case_ID)

    #base SRH-2D case directory
    base_case_dir = 'base_case'

    #name of directory for the new case, e.g., case_000001, case_000002, etc.
    case_name = 'case_'+str(case_ID).zfill(6)
    new_case_dir = 'cases/'+case_name

    #copy the base SRH-2D case folder
    destination = shutil.copytree(base_case_dir, new_case_dir, dirs_exist_ok=True)

    #chdir to the new case directory
    os.chdir(destination)

    #create a SRH-2D model instance
    if system_name == "Windows":
        #the following should be modified based on your installation of SRH-2D on Windows
        version = "3.6.5"
        srh_pre_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
        srh_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console_v365.exe"
        extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"
    elif system_name == "Linux":
        #the following should be modified based on your installation of SRH-2D on Linux (here we assume the srh2dpre and srh2d executables are in the case directory)
        version = "3.6.2"
        srh_pre_path = r"./srh2dpre"
        srh_path = r"./srh2d"
        extra_dll_path = r"./"
    else:
        raise ValueError("Unsupported operating system: " + system_name)

    #create a SRH-2D model instance
    my_srh_2d_model = pyHMT2D.SRH_2D.SRH_2D_Model(version, srh_pre_path,
                       srh_path, extra_dll_path, faceless=False)

    #initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    #open a SRH-2D project
    my_srh_2d_model.open_project(srhcontrol_file)

    #get the SRH-2D data
    my_srh_2d_data = my_srh_2d_model.get_simulation_case()
    
    #modify the inlet flow rate
    if len(Q_bc_IDs) > 0:
        inlet_q_bc_IDs = Q_bc_IDs
        new_inlet_q_values = Qs
        my_srh_2d_data.modify_InletQ(inlet_q_bc_IDs, new_inlet_q_values)

    #save the srhcontrol file after the updates of parameters
    if system_name == "Windows":
        my_srh_2d_data.srhhydro_obj.save_as()  #without any argument, the original filename will be used
    elif system_name == "Linux":
        my_srh_2d_data.srhsif_obj.save_as()    #without any argument, the original filename will be used

    #run SRH-2D Pre to preprocess the case
    bRunSuccessful = my_srh_2d_model.run_pre_model()

    #run the SRH-2D model's current project if SRH-2D Preprocessing is successful
    if bRunSuccessful:
        if system_name == "Windows":
            bRunSuccessful = my_srh_2d_model.run_model()
        elif system_name == "Linux":        #On Linux, the sleepTime and bShowProgress are not supported (there is some issue with the case_INF.DAT file)
            bRunSuccessful = my_srh_2d_model.run_model(sleepTime=10.0, bShowProgress=False)

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()

    #do postprocessing only if bRunSuccessful is true
    if bRunSuccessful:
        #convert SRH-2D result to VTK (This is hard-coded; needs to be changes for a specific case)
        my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data(srhcontrol_file)

        #wether the result is nodal or cell center
        bNodal = False

        if system_name == "Windows":
            #get case's base name 
            case_base_name = my_srh_2d_data.srhhydro_obj.srhhydro_content["Case"]
            if not bNodal:
                xmdf_file_name = case_base_name + "_XMDFC.h5"
            else:
                xmdf_file_name = case_base_name + "_XMDF.h5"

            #read the XMDF file
            my_srh_2d_data.readSRHXMDFFile(xmdf_file_name, bNodal)

            #export the XMDF data to VTK
            vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=True, dir='')
        elif system_name == "Linux":
            #read the SRHC files
            my_srh_2d_data.readSRHCFiles(my_srh_2d_data.srhsif_obj.srhsif_content["Case"])

            #export the SRHC data to VTK
            vtkFileNameList = my_srh_2d_data.outputSRHCDataToVTK(lastTimeStep=True, dir='')

        #create a directory for the vtk files if it does not exist
        if not os.path.exists("../vtks/"+case_name):
            os.makedirs("../vtks/"+case_name)

        #copy the vtk result file (only the last time step) to "cases" directory (one level above)
        shutil.copy(vtkFileNameList[-1], "../vtks/"+case_name+".vtk")

        #copy all the "Output_MISC" files to "cases" directory (one level above)
        for file in os.listdir("Output_MISC"):
            shutil.copy(os.path.join("Output_MISC", file), "../vtks/"+case_name+"/"+file)

        #copy the SIF file to "cases" directory (one level above)
        shutil.copy(srhcontrol_file, "../vtks/"+case_name+"/"+srhcontrol_file)

    # go back to the root
    os.chdir("../..")

    # delete the case folder
    if bDeleteCaseDir:
        shutil.rmtree(destination)

    #if successful, return case_ID; otherwise, return -case_ID
    if bRunSuccessful:
        return  case_ID
    else:
        return -case_ID

def run_one_SRH_2D_case_with_hydrograph(case_ID, srhcontrol_file, system_name, bDeleteCaseDir=True):
    """
    Run a single SRH-2D case with the specified case_ID. The ManningN value is used to set the Manning's n for the channel.
    The Q and WSE values are used to set the boundary conditions.

    Parameters
    ----------
    case_ID : int
        ID of the case to run
    srhcontrol_file : str
        Name of the SRH-2D control file (either srhhydro or SIF file), e.g., "case_SIF.dat" or "case.srhhydro"
    system_name : str
        Name of the system (Windows or Linux)
    bDeleteCaseDir : bool
        whether to delete case directory when simulation is done. Default is True. If it is set to False, the case directory
        will be kept. However, be cautious on this. If the number of simulations is large, all cases together will use
        very large space on the hard disk.

        In this example, the SRH-2D result is saved into a VTK file (thus the case directory is not needed after
        simulation is done).

    Returns
    -------

    """

    processID = multiprocessing.current_process()
    print(processID, ": running SRH-2D case: ", case_ID)

    #base SRH-2D case directory
    base_case_dir = 'base_case'

    #name of directory for the new case, e.g., case_000001, case_000002, etc.
    case_name = 'case_'+str(case_ID).zfill(6)
    new_case_dir = 'cases/'+case_name

    #copy the base SRH-2D case folder
    destination = shutil.copytree(base_case_dir, new_case_dir, dirs_exist_ok=True)

    #copy the hydrograph file to the new case directory (case ID is 1-based)
    source_file_hydrograph = os.path.join("hydrographs", "hydrograph_"+str(case_ID-1).zfill(4)+".xys")        # source file with path
    destination_file_hydrograph = os.path.join(destination, 'inflow_hydrograph.xys')     # destination file with new name

    shutil.copy(source_file_hydrograph, destination_file_hydrograph)

    #chdir to the new case directory
    os.chdir(destination)

    #create a SRH-2D model instance
    if system_name == "Windows":
        #the following should be modified based on your installation of SRH-2D on Windows
        version = "3.6.5"
        srh_pre_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH_Pre_Console.exe"
        srh_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe\SRH-2D_Console_v365.exe"
        extra_dll_path = r"C:\Program Files\SMS 13.3 64-bit\python\Lib\site-packages\srh2d_exe"
    elif system_name == "Linux":
        #the following should be modified based on your installation of SRH-2D on Linux (here we assume the srh2dpre and srh2d executables are in the case directory)
        version = "3.6.2"
        srh_pre_path = r"./srh2dpre"
        srh_path = r"./srh2d"
        extra_dll_path = r"./"
    else:
        raise ValueError("Unsupported operating system: " + system_name)

    #create a SRH-2D model instance
    my_srh_2d_model = pyHMT2D.SRH_2D.SRH_2D_Model(version, srh_pre_path,
                       srh_path, extra_dll_path, faceless=False)

    #initialize the SRH-2D model
    my_srh_2d_model.init_model()

    print("Hydraulic model name: ", my_srh_2d_model.getName())
    print("Hydraulic model version: ", my_srh_2d_model.getVersion())

    #open a SRH-2D project
    my_srh_2d_model.open_project(srhcontrol_file)

    #get the SRH-2D data
    my_srh_2d_data = my_srh_2d_model.get_simulation_case()

    #save the srhcontrol file after the updates of parameters
    if system_name == "Windows":
        my_srh_2d_data.srhhydro_obj.save_as()  #without any argument, the original filename will be used
    elif system_name == "Linux":
        my_srh_2d_data.srhsif_obj.save_as()    #without any argument, the original filename will be used

    #run SRH-2D Pre to preprocess the case
    bRunSuccessful = my_srh_2d_model.run_pre_model()

    #run the SRH-2D model's current project if SRH-2D Preprocessing is successful
    if bRunSuccessful:
        if system_name == "Windows":
            bRunSuccessful = my_srh_2d_model.run_model()
        elif system_name == "Linux":        #On Linux, the sleepTime and bShowProgress are not supported (there is some issue with the case_INF.DAT file)
            bRunSuccessful = my_srh_2d_model.run_model(sleepTime=10.0, bShowProgress=False)

    #close the SRH-2D project
    my_srh_2d_model.close_project()

    #quit SRH-2D
    my_srh_2d_model.exit_model()

    #do postprocessing only if bRunSuccessful is true
    if bRunSuccessful:
        #convert SRH-2D result to VTK (This is hard-coded; needs to be changes for a specific case)
        my_srh_2d_data = pyHMT2D.SRH_2D.SRH_2D_Data(srhcontrol_file)

        #wether the result is nodal or cell center
        bNodal = False

        if system_name == "Windows":
            #get case's base name 
            case_base_name = my_srh_2d_data.srhhydro_obj.srhhydro_content["Case"]
            if not bNodal:
                xmdf_file_name = case_base_name + "_XMDFC.h5"
            else:
                xmdf_file_name = case_base_name + "_XMDF.h5"

            #read the XMDF file
            my_srh_2d_data.readSRHXMDFFile(xmdf_file_name, bNodal)

            #export the XMDF data to VTK
            vtkFileNameList = my_srh_2d_data.outputXMDFDataToVTK(bNodal, lastTimeStep=False, dir='')
        elif system_name == "Linux":
            #read the SRHC files
            my_srh_2d_data.readSRHCFiles(my_srh_2d_data.srhsif_obj.srhsif_content["Case"])

            #export the SRHC data to VTK
            vtkFileNameList = my_srh_2d_data.outputSRHCDataToVTK(lastTimeStep=False, dir='')

        #create a directory for the vtk files if it does not exist
        if not os.path.exists("../vtks/"+case_name):
            os.makedirs("../vtks/"+case_name)

        #copy the vtk result files (all time steps) to "cases/vtks" directory (one level above)
        for vtkFileName in vtkFileNameList:
            shutil.copy(vtkFileName, "../vtks/"+case_name+"/"+os.path.basename(vtkFileName))

    # go back to the root
    os.chdir("../..")

    # delete the case folder
    if bDeleteCaseDir:
        shutil.rmtree(destination)

    #if successful, return case_ID; otherwise, return -case_ID
    if bRunSuccessful:
        return  case_ID
    else:
        return -case_ID
