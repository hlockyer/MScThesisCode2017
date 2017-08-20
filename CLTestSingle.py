#!/usr/bin/env python

"""\
%prog [options] <yodafile1>

Run a test on a specified yoda file, returns limits and plots found
against any validated contur analyses.
"""

from contur import TestingFunctions as ctr
from contur import Utils as util
import rivet
import yoda
import sys
import os
from optparse import OptionParser

parser = OptionParser(usage=__doc__)
parser.add_option("-o", "--outputdir", dest="OUTPUTDIR",
                  default="./plots", help="Specify output directory for output plots. \n")
parser.add_option("-g", "--grid-mode", dest="GRID_MODE", action="store_true",
                  default=False, help="Run in gridmode expects a prearranged grid of yoda files as input[DEPRECATED].")
parser.add_option("-p", "--make-plots", dest="MAKE_PLOTS", action="store_true",
                  default=False, help="Draw ratio plots.")
parser.add_option("-a", "--analysisdir", dest="ANALYSISDIR",
                  default="./Analysis", help="Output directory for analysis cards.")
parser.add_option("--hier-out", action="store_true", dest="HIER_OUTPUT", default=False,
                  help="write output dat files into a directory hierarchy which matches the analysis paths")
opts, yodafiles = parser.parse_args()

if not yodafiles and not opts.GRID_MODE:
    print "Error: You need to specify some YODA files to be plotted!"
    sys.exit(1)

class Plot(dict):
    "A tiny Plot object to help writing out the head in the .dat file"

    def __repr__(self):
        return "# BEGIN PLOT\n" + "\n".join("%s=%s" % (k, v) for k, v in self.iteritems()) + "\n# END PLOT\n\n"



if __name__ == '__main__':
    # Command line parsing
    opts, args = parser.parse_args()

    for f in args:
        if not os.access(f, os.R_OK):
            print "Error: cannot read from %s" % f
            sys.exit(1)

    for infile in args:

    # need an empty dict to store our results
        scatterpoints = {}
        masterDict = {}
        heatMap = {}

        plotdirs = [os.path.abspath(os.path.dirname(f)) for f in infile]
        plotparser = rivet.mkStdPlotParser(plotdirs, )
        #plotparser = util.mkStdPlotParser(plotdirs, )
        for anatype in ctr.anapool:
            masterDict[anatype] = []
        refhistos, mchistos, xsec, Nev = util.getHistos(infile)
        hpaths = []
        for aos in mchistos.values():
            for p in aos.keys():
                if p and p not in hpaths:
                    hpaths.append(p)

        util.getRivetRefData(refhistos)

        mapPoints = {}

        for h in hpaths:
            print h
            try:
                refdata = refhistos['/REF%s' % h]
            except KeyError:
                sys.stderr.write('Ignoring %s, no refdata found.\n'%h)
                continue

            # Manually store additional plot in a function called LumiFinder, if a Lumi isn't stored vs an
            # analysis name then use that info to veto testing
            # LumiFinder also returns -1 for vetoed/blacklisted histograms
            if ctr.LumiFinder(h)[0] == -1:
                continue

            lumi = ctr.LumiFinder(h)[0]
            if lumi > 0:

                # Use this switch to view individual analyses
                # if '/ATLAS_2014_I1279489' not in h:
                #    continue
                mcpath='/'+infile
                if mchistos[infile][h].type == 'Histo1D':
                    # Most of the time we deal with 1D histos 
                    has1D = True
                    mc1D = mchistos[infile][h]
                    mc1D.setAnnotation('Path', mcpath + h)
                    sighisto = yoda.core.mkScatter(mchistos[infile][h])
                else:
                    has1D = False
                    mchistos[infile][h].setAnnotation('Path', mcpath + h) 
                    sighisto = mchistos[infile][h]

                # some special logic to deal with normalisation
                normFacSig = 0.0
                normFacRef = 0.0
                if ctr.isNorm(h)[0] == True:

                    if has1D == False:
                        print 'Found a normalised 2D scatter. Makes no sense. Ignoring it.'
                        print h
                        continue

                    for point in refdata.points:
                        normFacRef += point.y
                    for point in sighisto.points:
                        normFacSig += point.y
                    normFacRef = ctr.isNorm(h)[1]
                    import numpy as np
                    if mc1D.sumW2() == 0:
                        continue
                    normFacSig = (float(mc1D.numEntries(
                    )) / float(Nev.numEntries()) * float(xsec.points[0].x))

                
                if has1D:
                    if (mc1D.sumW()>0):
                        mclumi = float(mc1D.numEntries())/mc1D.sumW()
                        if (lumi/mclumi>2.0):
                            # Note, this is usually in pb-1, but sometimes in fb-1.
                            # Also, some analyses have factor of 2 because they are averaged over e and mu channels
                            # How this works:
                            # 1) Rivet scales the histos by xsec-per-event
                            # 2) sumW = n_Gen * xsec-per-event = xsec
                            # 3) so n_Gen/sumW = n_Gen/xsec = L_gen = mclumi
                            print 'Warning! Effective MC lumi %.2f is substantially less than data lumi %.2f for %s' % (mclumi, lumi, h)
                            print '--> consider generating more events.'


                CLs = []
                sigCount = []
                bgCount = []
                bgError = []
                sigError = []
                # fill test results for each bin
                # out=np.zeros([refdata.numPoints,1])
                for i in range(0, refdata.numPoints):
                    global mu_test
                    mu_test = 1
                    mu_hat = 0
                    varmat = 0
                    # Sigerror is used to store \tau, the ratio of MC Nev to "data" Nev
                    # TODO check! (JMB) - looks to me like it stores the generated luminosity, not the ratio.
                    if ctr.isNorm(h)[0] == True:
                        sigCount.append(mc1D.bins[i].sumW * lumi * normFacSig)
                        bgCount.append(refdata.points[i].y * lumi * normFacRef * (refdata.points[i].xMax - refdata.points[i].xMin))
                        bgError.append(refdata.points[i].yErrs[1] * lumi * normFacRef * (refdata.points[i].xMax - refdata.points[i].xMin))
                        if mc1D.sumW() ==0:
                            sigError.append(0.0)
                        else:
                            # TODO: shouldn't this be adjusted for normalised histos?
                            sigError.append(float(mc1D.numEntries())/mc1D.sumW())
                    else:
                        if has1D:
                            # error on signal will be estimated from event count, using poisson stats
                            sigCount.append(mc1D.bins[i].sumW * lumi)
                            bgCount.append(refdata.points[i].y * lumi * (refdata.points[i].xMax - refdata.points[i].xMin))
                            bgError.append(refdata.points[i].yErrs[1] * lumi * (refdata.points[i].xMax - refdata.points[i].xMin))
                            if mc1D.sumW() ==0:
                                sigError.append(0.0)
                            else:
                                sigError.append(mclumi)


                        else:
                            # TODO: need to use the uncertainty on the 2D plot somehow.
                            sigCount.append(sighisto.points[i].y)
                            bgCount.append(refdata.points[i].y * (refdata.points[i].xMax - refdata.points[i].xMin))
                            bgError.append(refdata.points[i].yErrs[1] * (refdata.points[i].xMax - refdata.points[i].xMin))
                            # Sigerror is used to store mclumi, but when the MC plot is a 2D scatter, this is not relevant and is set to 1.
                            sigError.append(1.0)



                    # cater for the case where the refdata bin is empty,
                    # occurs notably in ATLAS_2014_I1307243
                    if refdata.points[i].y > 0:
                        CLs.append(ctr.confLevel([sigCount[i]], [bgCount[i]], [bgError[i]], [sigError[i]]))
                    else:
                        print 'Warning! Ref data bin '+str(i)+" empty in "+h
                        CLs.append(0)
                anaobjects = []


       ## DrawOnly is needed to keep the order in the Legend equal to the
        ## order of the files on the command line
                drawonly = []

        ## Check if we have reference data for the histogram
                ratioreference = None
                if refhistos.has_key('/REF' + h):
                    refdata = refhistos['/REF' + h]
                    refdata.setAnnotation('ErrorBars', '1')
                    refdata.setAnnotation('PolyMarker', '*')
                    refdata.setAnnotation('ConnectBins', '0')
                    refdata.setAnnotation('Title', 'Data')
                    #if ctr.isNorm(h)[0] == True:
                    #    refdata.setAnnotation('Scale', str(normFacRef))
                    #if opts.RATIO:
                    #    ratioreference = '/REF'+h
                    anaobjects.append(refdata)
                    drawonly.append('/REF' + h)
                    
                drawonly.append(mcpath + h)
                #if opts.RATIO and ratioreference is None:
                #    ratioreference = mcpath + h


                # if opts.RATIO and len(drawonly) > 1:
                #     plot['RatioPlot'] = '1'
                #     plot['RatioPlotReference'] = ratioreference

                for i in range(0, refdata.numPoints):
                    if ctr.isNorm(h)[0] == True:
                        sighisto.points[i].y=(sighisto.points[i].y*normFacSig+refdata.points[i].y*normFacRef)*1/normFacRef
                        sighisto.points[i].yErrs =((refdata.points[i].yErrs[1])**2 + (sighisto.points[i].yErrs[1])**2 )**0.5
                    elif has1D:
                        sighisto.points[i].y=sighisto.points[i].y+refdata.points[i].y
                        sighisto.points[i].yErrs =((refdata.points[i].yErrs[1])**2 + (sighisto.points[i].yErrs[1])**2 )**0.5
                    # if there was no 1D plot, assume we had an already-made ratio (ATLAS MET)


                    # write the bin number of the most significant bin, and the bin number for the plot legend
                    sighisto.title='[%s] %s' % ( CLs.index(max(CLs))+1, max(CLs) )
                sighisto.setAnnotation('LineColor', 'red')
                anaobjects.append(sighisto)
                plot = Plot()
                plot['DrawOnly'] = ' '.join(drawonly).strip()
                plot['Legend'] = '1'
                plot['MainPlot'] = '1'
                plot['RatioPlotYMin'] = '1'
                plot['LogY'] = '1'
                plot['RatioPlot'] = '1'
                for key, val in plotparser.getHeaders(h).iteritems():
                    plot[key] = val
                ratioreference = '/REF'+h
                plot['RatioPlotReference'] = ratioreference
                output = ''
                output += str(plot)
                from cStringIO import StringIO
                sio = StringIO()
                yoda.writeFLAT(anaobjects, sio)
                output += sio.getvalue()
                util.writeOutput2(output, h, opts.OUTPUTDIR)
    # All these extra count checks are to stop any plots with no count in most likely bin from being entered
    # into liklihood calc, should be fixed upstream

    # See if any additional grouping is needed 'subpool'
            max_cl = CLs.index(max(CLs))
            if ctr.LumiFinder(h)[2]:
                tempKey = h.split('/')[1] + '_' + ctr.LumiFinder(h)[2]
                if tempKey not in mapPoints and bgCount[max_cl] > 0.0:
                    mapPoints[tempKey] = [
                        float(max(CLs)), 
                        [ sigCount[max_cl] ],
                        [ bgCount[max_cl]  ], 
                        [ bgError[max_cl]  ], 
                        [ sigError[max_cl] ],
                        str(h)
                    ]
                elif bgCount[max_cl] > 0.0:
                    mapPoints[tempKey][1].append(sigCount[max_cl])
                    mapPoints[tempKey][2].append(bgCount[max_cl])
                    mapPoints[tempKey][3].append(bgError[max_cl])
                    mapPoints[tempKey][4].append(sigError[max_cl])
                    mapPoints[tempKey][5] += "," + (str(h))
            else:
                if h not in mapPoints and bgCount[max_cl] > 0.0:
                    mapPoints[h] = [
                        float(max(CLs)), 
                        [ sigCount[max_cl] ], 
                        [ bgCount[max_cl]  ], 
                        [ bgError[max_cl]  ], 
                        [ sigError[max_cl] ], 
                        str(h)
                    ]
    # Scan through all points and fill each category, stored in masterDict,
    # with the counts from each grouping if it is more sensitive than the
    # previous fill
        for key, pts in mapPoints.iteritems():
            poolname = ctr.LumiFinder(key)[1]
            pts[0] = ctr.confLevel(pts[1], pts[2], pts[3],pts[4])
            if not masterDict[poolname]:
                masterDict[poolname].append(pts[:])
            else:
                for listelement in masterDict[poolname]:
                    if pts[0] > listelement[0]:
                        masterDict[poolname][masterDict[poolname].index(listelement)] = pts[:]
    import pickle
    # print everything out


    sigfinal=[]
    bgfinal=[]
    bgerrfinal=[]
    sigerrfinal=[]
    for key in masterDict:
        if masterDict[key]:
            sigfinal.extend(map(list,zip(*masterDict[key])[1])[0])
            bgfinal.extend(map(list,zip(*masterDict[key])[2])[0])
            bgerrfinal.extend(map(list,zip(*masterDict[key])[3])[0])
            sigerrfinal.extend(map(list,zip(*masterDict[key])[4])[0])
            masterDict[key].sort(key=lambda x: x[0])
            util.writeOutput(masterDict[key], key + ".dat")
            with open("./ANALYSIS/" + key + '.map', 'w') as f:
                pickle.dump(masterDict[key], f)

    if len(sigfinal)>0:

        sumfn = open("./ANALYSIS/Summary.txt", 'w')
        pccl = 100.*ctr.confLevel(sigfinal, bgfinal, bgerrfinal,sigerrfinal)
        result = "Combined CL exclusion for these plots is %.1f %%" % pccl
        sumfn.write(result)
        for anapool in masterDict:
            if masterDict[anapool]:
                sumfn.write("\n"+anapool)
                sumfn.write("\n"+str(masterDict[anapool][0][5]))
                                                                                             
        sumfn.close()
        print(result)
        print "Based on " +str(len(sigfinal))+ " found counting tests"
        print "\nMore details output to ANALYSIS folder"
    else:
        print "Contur found no analyses it understands in the file"



# print 'Run finished, analysis output to folder "ANALYSIS"'
