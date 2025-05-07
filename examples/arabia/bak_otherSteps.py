
# In[]:
###########################################
#       Step 2 : est variogram as Cp
###########################################
flag.run_Cp = False
if flag.run_Cp:

    # variogram
    insarDeramp, Cds_set = calc_insar_variogram(dataDict, out_dir=out_dir, distmax=300., rampEst=True)

    # plot
    plot_all_variograms(dataDict, insarDeramp, out_dir, savefig=out_dir/f'semivariograms.pdf')
    plot_all_variograms_image(dataDict, insarDeramp, out_dir, savefig=out_dir/f'deramp.png')

    # save Cd, Cp into file
    report_avg_covariance(dataDict, Cds_set, out_dir/f'Cchi.txt', sk=20)

    # inversion with Cd + Cp
    blk_Cp = run_inversion_Cds(dataDict, blk, flag, Cds_set, savefig=out_dir/f'Cp.pdf')

    plot_geodesic_profile(flag, blk, dataDict, pole, poleA, gpsGDf, gpsGDf_full, fname='geodesic_profile_Cp.png')


# In[]:
###########################################
#       Step 3 : iterative Cp inversion
###########################################

if flag.run_IRLS:
    # iterative residual error
    if flag.run_Cp: Cds_set = Cds_set
    else:           Cds_set = [0] * len(blk.res_set)

    for i_itr in range(int(flag.run_IRLS)):
        # remember the old model params
        mp_old = np.array(blk.m_all)

        # update Cp with residual error
        Cds_set_irls = []
        for j, (res, Cp) in enumerate(zip(blk.res_set, Cds_set)):
            res = res[blk.roi_set[j]]
            Cr = np.diag((res)**2)
            Cds_set_irls.append( Cp + Cr )

        # inversion with Cd + updated Cp
        blk = run_inversion_Cds(dataDict, blk, flag, Cds_set_irls, inv_opt=3, sk=20, savefig=out_dir/f'Cp_itr{i_itr+1}.pdf')

        # report convergence
        mp_new = np.array(blk.m_all)
        learn_rate = np.linalg.norm((mp_new - mp_old)) / (1 + np.linalg.norm(mp_new))
        print(f'#> {i_itr+1} iteration: Cp += R^2, learningRate={learn_rate}')

        # save Cd, Cp into file
        report_avg_covariance(dataDict, Cds_set_irls, out_dir/f'itr{i_itr+1}.txt', sk=20)



# In[]:

if False:
    ## Final plot
    # update the block inverted pole
    pole = postfit_outputs(blk, dataDict, flag, pole_ref=poleA, extname='Cchi')
    poleDict['InSAR'] = [pole, 'b']

    plot_globes(poleDict, flag, savefigs = [
                                out_dir/f'pmm_est_Cchi.pdf',
                                out_dir/f'pmms0_Cchi.pdf',
                                out_dir/f'pmms_Cchi.pdf',
                                ])


    plot_pole_motion(poleDict, flag, frames=['ITRF2014','Nubia'], savefig=out_dir/f'poleVector_Cchi.pdf')


# %%
