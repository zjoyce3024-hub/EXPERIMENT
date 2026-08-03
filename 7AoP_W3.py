#!/usr/bin/env python3
"""Compare BOME, Sequential QCQP, and PROBE using embedded loss data.

The script is self-contained: all runs needed for the figures are embedded
below. Running it creates these files beside this script:

    train_loss_bome_qcqp_probe.pdf
    validation_loss_bome_qcqp_probe.pdf

Dependencies: numpy and matplotlib.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import tempfile
import zlib
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "method_comparison_matplotlib_cache"),
)

try:
    import matplotlib
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Install it with: pip install numpy matplotlib"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_LOSS_INDEX = 0
VALIDATION_LOSS_INDEX = 1
METHODS = ("BOME", "Sequential QCQP", "PROBE")
METHOD_SLICES = {
    "BOME": slice(0, 10),
    "Sequential QCQP": slice(10, 20),
    "PROBE": slice(20, 30),
}
COLORS = {
    "BOME": "#0072B2",
    "Sequential QCQP": "#E69F00",
    "PROBE": "#D55E00",
}
MARKERS = {
    "BOME": "o",
    "Sequential QCQP": "s",
    "PROBE": "^",
}

# Layout: [metric (train, validation), run (10 BOME + 10 QCQP + 10 PROBE),
#          iteration].
_DATA_SHAPE = (2, 30, 100)
_DATA_DTYPE = np.dtype("<f8")
_DATA_SHA256 = "ab5d070d0a00350de71bee96e9f13ed3cd71073912f36e0c77ace1029480878e"
_DATA_B85 = (
    "c-n=1d7Ku-wRVf_n=B&Rz_cQ+GYtA^G%8MKF}ZPxOT^shjnit>#JC%ci3;VKMT~pgp@F!fpopl5Oo42ofFL0IG$<nbGQ&FC$F1i%"
    "eW}CS$^28l>bJVOy6QaVJm*yPnIH(FWyN6>?E8DFvX?7=U0M9UWqLg7_TZ}C?)uW+?z)*V$koG1E8Xw$o0Cu<h3moSLGF6=LW8TP"
    "@2o6x^+#8_`|a~9-Szk^JU4vva@Rh21NWKaa`(G^brR}tH@N$gmxj9QW@K;N{~Y=cpBU=;jb6k!f~PNc_2jKHy?%ocXX3tBFm7ol"
    "{mbvt-i{kuwEy_z_wd6r-Szn0Nk#kLZ{NQh{Y>rMYybBSCN}hR{Z8nO{-*bK^>ERo?yes@)YZ+bNj+S<1uJoV_8?aew+`y+`uX<q"
    "ZmwQ``C+bJeP&nh_jZkkx_Z}QSK3ZhU0lEIt6V+l*4WuyKcd)`L60g|#>ZE5a@WfmJG%0Wst&IGiIuJ%pIYu+kIt+_d3s|9uInN0"
    "_eUBI;kvZv`g-m6ty_DpYdg+kTdsF|&U-t~M;mwF>qFadUR}HJ%Z65*w^m-e@q^m#-+!`dQcJH~u+p`U);F}^yty)&(a_qJn}@o3"
    "I_Gj%HwQH~bNx51bmhJ~-MUBh<tS@`hj4Et<Z18VUZ@Z2?U#eSda$>=7-ccbisJp-MSI!*-YVc~Z+Wl%um4|vRbIPvPZj$c47uN1"
    "4msP~*xT=?s@MMgn7xC6%e{^K9e6i;8+m^PpHs~CRYm*vm+Y+oF83DmdNHpzI(fExiy`N=<xZZ{y^T(ugWB?*s1JsG*ADKEvI25n"
    "Tik8`($o&_ih3i<3U@uJEp~NVTLC=OLC(|KVNUMDI^ZCxtLTh!aOeH&aosQ{_eotvN0ftIY3c^M>uKF!C%3i^a-Y^!bl|+U=eoDU"
    "_2Lfum$nY_p43&LzrDlS?%&_O^#@+|mbc-&wD$U8-R>RS#xE<{aK2ihUdei8E6#hH{hw!RD_cNbiy{BDjV(E!EpY#^Hk{WMobQ&n"
    "uedqpsp?>~ABNxSiczntXoC42+|<>#4RdlAuSOXyhdhRhAt!b|<S6_crI}8-nlTK&Plwz^Qz5Tr<}k?RtV-y`Pk<XUg?4E=@Rm-c"
    "9P_%gvoi-%&k{%MKb(f=CR2&eslZV>t<n2Eo>5-7&i>^2@l@bHnFf4>Gb(!R-=Ex<%m6L|zF+Vua1(z%%*ns`wAg!JJ8LlTF?$&8"
    "0!sNkom~Z-%o#>IQ4CzoEAIw91|031cZUx%pZHlc7`Rvjd?X8DAEK{dH-e?ZI`3bmi^^d?7FXc+CBT!^!{xw<l<^A4dsGFy#4CZT"
    "a0TqjzIAePAAB;{mC=+cuO7^-YP)~Ced`N*@hRkb-@1AAee>C#^VNpy?(R>f4}*P~16=HzABU&tGvF=!ywcT!&wwW>v**bDbIZN^"
    "lP{~9b6$amg|rha!-c?2RxYnR81^FozY0REe-NuO>7}l-ZhNb;nD(gH@6W1w%KJl>Y482**naI<2EB9YGLByAch-%JyU2fj)RXOa"
    "-SqJ9v;5u2?>+r8Wc!Gv?XJphEW5Iul!x&-ER#dot_$aZ^@MfXnfI|Y{5|Z%>nzg_{Jo=p-SB$MGE#N<J#5e4rEJS_9>Q_3ZrkyG"
    "woBUj?b9~wkL?21ZEHS{Wx)1fEB42F%=QV}r7d{B^wXUENIT&_&HQ@M!mpdAY}b<4Dfd>&CTtgQ9Ii|nD+49ZO1?#IwLGQ@H<89g"
    "FJImh!Ii>^DfaP}sQ%L;zn`Gk$AidocD+bl@8yqQewXJ7K2n7Pxh|zu_>g)kxMG<SA2IO}D!c?;{W2s@Ocx(N_AnnGf(xm~s?2ag"
    "d<3jp!9zzMKZ1*t@^1tetcP-+$ax396#0)84y10{`8W_f3C;|05lN|V6bqhO`oBv(ZSLdAvL7pWY3k!9ZB`f;<{<XN_D10)41BvF"
    "cEE_9?d8k0RdO0Cy`8B0LWQ$1NAK%0vs+5uGyEw01?qaL`U#7Cdk`sHBnmHr19^Td^`3q`vpYR~90*RNOvGLYKG@FW;3F10aj0)U"
    "GJJIMuP5?*CxxdDh3$(y#K%db@zY+}pZ5OsMByM6dq8;&TKjrFYOV0pO38Q2!gfXUy=md&M(_|f_wge3M(Uy9q=m9G3O`vtO?_OL"
    "#&S>2q8$Dr@|Gz4h&*RFQe_w_JQNky`;n4&jkgRBMTPsZ>i#GPPhn4GmlV&__QLk?aUpg>_?6&9c$K02r;1mFUH#u<;vf}Vunf8="
    "Ty*l~KJE1H?M2Yhw-=)Cla9V!kTRvbChdKDBKQ#=CcH|17d|HVh}!wM2wVHSNUjV2%6OQv7pdT?t&f|ix&OP!v)GG(?Lxr^?LgYZ"
    "$Aj31NZF5&?JVm_6W{Jw&3B4@{c99YNflp86kfDn5-a;5{!WnN&!nn9Ezd?>Pu2J`zd`Y}K;NhEk}7<N{7bu}yU(j4g&X-@O7Tww"
    "7s8vwZit^DJS*+&`yoczhg9(@!AEqcU(fK-(Z8PYEhYcqA-=zn*_HOb-WPr(evIgQtL#K5{)pI{R=%GSDLlw^t9cdcNekaT2tESE"
    "*Nn0QL37`JWIU=#Vf&NW6~RZ~>tm5Ko8w;x%8wQ~x9Ymq+eYCjQn-lKbF8*!y_Dag@s@+5K-m?ImsHt5!DFKFBlysKO5sF)kCeYI"
    "cnOq$qWyO9(_`_^g=Z=MJrMkK^6gE=kK}qspO1yg-h_%@Nj>S{%eUYxP=18?E7C4jaxS<LyaerhJS1&=oQPc!za&-o5PK50P&}-;"
    "Z-?Z%!~?9d9~mDz*tZv9GanzqvjSx=GXGrpvqsBjufp*tt8fx(KVSLZM)}>QsPKI{j+Sdrl%J9L5!$cO{<?a8s{J3ur^LSq6%OS0"
    "u$%9fB;9<wA%2JWC1M8z7lMaa$FWr1c2xdHM_<k}-lgMK>i5hpv{(3Q>)RdSWtlvSKXQnVgSeHCpGf)B;)kRx!<N1uBmRl_AF1M7"
    "f$}TFPKf;poBO{@|AH6cW5TO4f5WpMX(RR%l19e;8@aF0fPK5TfpLG<qk8<F)L~yDt*tEL{-U!dW)Joi(%sx|+l6+?4(u1EHQdkJ"
    "j(rxh4bKg>Vqd~+#y&)_0l(XI=r3A}`-5*Wj&wEl3&U01&rETDv;zAX=~9%*dW<VwfPTyu+=rQqe#~5q)2to_dpHfxu`{p_lT1cG"
    "wi5e2!3TJ5Jf814j{7g8u`imA=DyG?*w2Ya^7?S>$D}V}ymlz}Ri4E2qGvgd$IxFog!?sy`%(8{zsL^aKGdD)$K1+&qj5No5WQ33"
    "%J_QRpNztL*(><`U*o;)#oWgoh<fsC?0Y6>@%;zzxj*5#igL6IBCJpRJ<jt9DB}{!$8j8YH|(dG5btB|$GC!?+<)qg`L%~)9!$fT"
    "cwSSs-#Q87+cA{#iSr1ydMMtj8s`+kZMYsU!MTidEzfH#z&VU$A<kjMmqHHgT%L#c1m`f)$@o1OkM_aGICl|`y4>SEwvqBgd_;{{"
    "=d^+QXbr$o#JV9)f_m=%)p4Hdu}+rz*a7#c?QZr{i{I0ooQEC2UsBzOaa42vaU0s(Exd0N<~!KLeem_T9;^r6g0;MF4RO8|>tU-P"
    "2hmE()k^H=g-eM4Mcj8>K)Ig7ebRYc?@#f5=?vm&8lE3d;&?tHPCp<H#&Vy35zo(z=04@ic)#!!K6fPFV<_?U9Od>&>Z`|Cf0*S%"
    "l+V9HAEkF;ev|j{ocQ<DC%?lu%vC(k@LR}paw%~+kmn*UAimGSd!#RM-O6!2JQ3rK`V+s$;=O{t#Ql+6-=iS!Nqam$X-Ql)Lwj3~"
    "^FLw0ecjt3r|~YF*D=)%&}$nhuWL|73xSt-1?77-&krrY`JeE^LFoS?th4zT=Z2y&z*961I0_Qr#y*ex%`-R$6%PeY;=wp)6g`al"
    "?E@&y?|DwC5%U=nA1QHQ8i4P8b_3(7hu%%<P>*XdZ!yo2*gd$<?!vq#JBgzmkh^#f?aMaG(H83c&6K+h9LGA8$+vi3{0-&rYv`q@"
    "3hR?Br`#^bbK}L_2VVfZ+4*Q6&`#J{#PclLDejY-snFZeB)<Pd=(A`%<$5gfGn#hm4L;{J;%PSZ_6t1k@*?fuKY6a@Db&Lu#OGhA"
    "FAU#*F#3z`!MK7uSihO=ep@9a^wp)r<*#VhFGd;u9P<|bnDhSw%v*Xs_WPrs@f=V&?QB2HUs%fddw4MPRv(VD7tc5J!FimZHJ^JZ"
    "*RvVzdK<oP9narY^L*Wwp<G|cd$JTbNmc<D=_=ZR#lT1M3FJNb73Ji8)Z>qN9&aq={A`RfB5o|@Kj!&7`#8=CMvnmx(PKP!_a{8x"
    "oJ@UkC(jLD*?{N99N)D(=XVXy`H^3R#7EEw`G_0Hhw4QSaQ*6uhdQof9eF`5&yDarmYmy)$hV^1SdVBIdD1T0<(;@c+(CJ+p*^pr"
    "Ufo8$wH2k=!gD8LPd3tyZ=il%$NG1aueFrVHGI#n$tzZ29Ht8Lm9C`SSwXp9LYyrEPJ^#FzWLCP={)is;c0VlpV*st2K58axula>"
    "UxD?HKBm3?2=ip$#rs9$@Er3N^~flm?|B{LvoDi34W*rbj&lASe}9VoK1sdyH}d5N`1}XCE_YM@@1S10o9lTE?a&ppqt_4*S8$w{"
    "P+$C<cKsZlpBsSo;W^|@CzJn`Q7@f9yL&u&>fv1fw!m}TmFI{Kqx?0ceQXVUrt2ujH}gF0CdzLW@DVPByxY%!tMm)vuo8ACnh88b"
    "ydF)MB<D`?zGKL{o<u!)YLLgT(ni|hM%w?zM!^k!Z{T<vD4z{DXC=4^>&cJm$iwQ`zLx8@mwJu3v3sb$cSHWd-Q-6ZZm7R^a2;z{"
    "R?{EYM*Fgj=L@$`j<@if<0j&16Y;l!<66h->nK0pQV!OT&wWjPvx>YdWqk$xj^$YIcp1m{72YddOg*}gyzfigx6jYPxaLsb&!v5u"
    "MSVV<c$V{BQ)q`KbABc^ay`k1##7$M@%e8PCvVd3zDj*I68$Ex;r-KBaIVvipgnn-cKB(Y>m;9x9<A{Fo_)B&cqahY;qByaH*=i7"
    "BmcUdeDQZYxArUQL;5+<xtzaWaNf_Q+?)lxgeP(xPQrTyNArAZ5#~2Ig7&))=FfJa9dAduIhga_gz{6(^W810aQ<y0aA8-`uC1Vd"
    "GH)e$G}=e+H*mdrd+{LKNd83sL-<xm`;rn5rh)Qa&v~lneAaQj>bS18T+i%0F7?wM$ZstE2>thHC;8eA^1B-Hb>bn~PQAXBJYzHU"
    ";b!7t1MSH=^5*Yo$JXGzqOZB$l>4x%((_Bw<y^0&#Q!3W?@RLWFDvMuP@m6YKQk%knS4{uCen^fpdF)qNZw(8?@%ATA^r&4597SP"
    "$a#8(cJC?b=|}13J;ZjC=+E85@&1u|?N*NOI^yp-j^k?fa}DL}YR>OPl&=eEw|~lYKaDv1FW80jRQ#SCOCC^4dFVwv90PqH^<jLb"
    "6YY9)o?qPqzdPJP|Di2;^JeJ(WDVpr-b(-b8~Pv1;TME6;1>jz<z&cvIum}i8H4-n+rUHcBAyeDWPJpA*(2m@57Do`1-P(xq8{8y"
    "{4~;^$aog*0C5mC&`+;tyL!BLQb#_J*$LvlmUdLm#qZ(yT=J`Q7wz*->RraQk{bGn)#PK<#KAVo|2E=d3)g2e`OYT#i5rRM^<0N_"
    ")QjIyuYF59^9|R1746Mx_Op`jy@GPNl=hB1E7(9eTu8rR0qxmZ#znp$Kbu2-Lw_fpMg2IHa%PG9mGCRWiS(~3$y48F|L+s0^nao^"
    "=}){tz3>|4{8i5LNIvIz@~!8oryi$%AHwH923$seNB{PI>g5M$XZ}Qd-%dMy6Y+Zu=k*H4FRwxS;MaWarQ}8D@_U9K^F1Qe(*d04"
    "AJLALa{hXe&y~;*>_K^I4}9Auj5qGZ_d{_V<#{*wPgKM2cGl4ESUrh+nesgse96p{couLJPB@eC6voX)^LwQ?=zl!fKwic8191`4"
    "|419C_ZuoCF2#72_+yltTKVpSdS?&$?JmmSPRdCQ%WBFid6C_W`H41>M{J;et${q6wT%CM%ee4r^4XNQSV3O2g!8!s`pYil`qNKO"
    "<`VC-Xt!sQAB*0fhW8FDiSrK_k6FTTEJB<#evABU4A1|+0r@Z^*<a?bbN)W%{ftWmPcyFj2;%_{&<@;3yEz2kkJ#JEM{eTxFSnDo"
    "-$uQEJ=g6juFqBU|1PC{xeVh=&*i+HOF!ux`V;3+zR#dO7{LBc<oh2(zIGgOb~L}2E28|C;Jkj^m2%&adaD)j)kxlQ5bxhfJ-?aP"
    "*Fv7{*Qh7gLeB(@(*{>Do;HVZr)m5??=$%G@p#yU=p*1HnQ|uW3GaWMaz1LM|9#wZ{Jw7p`b+*s|NcqD3!=ZV|3A~eCmw7A<(m5("
    "rk?&M{qTK$1nT_GXdh3)_;%C3BM#Ec4p1*`WBhij;DLH}BkctBefS;abUWweYwoY4v<p=gqUUk{VvcVS*L5D{U^ea5XVi0_a{SY1"
    "hbGe>x0LITBu+|w_MZ61<ngalZ~u#aQbId3ociKf;^+nXRnOADc$D)pgnXEBtH6-=e8f1)9mMsG<a@VKfBc4g@CM-BT*)}xZ^&Q%"
    "hwaWIj?UpcN5oS^zwtTxvk`H9GVRg#iPPg~Pm4Hj#l&gI{(Epf+EZ_}pdGoB@vc48!<)$4x6@y)?#*=rpR(J)m+WH7@f!HU_6pi%"
    "@_##-{Y`2hZ-u;?@#J4)Dfbh=ui}Jp;NgtxzJY$z$9exy#)A!c);+WrcVnF9A>hLPg}m-2;K$y1xp)4;HZq>0{Bqksd(C}9TTeSt"
    "Fa8Gk5&4zj_k(s1dHrtgC+(_~coph3v^&*|Cv7L5w(vQOPo*2lPl=;oJ-@&Dmh!inc5ju~721hq+<&B9F<;RiA)eB?wCl4eN3+P^"
    "XHx%G(Oy{Mcru<V`y9!;eDATu{af@)#_+qvmx;TCae-mfD=!e2k8@w|6Uy&jIByTpUfx4KKbUr7Jo(t2z(;Z|_0x5f-z$l~%NP&-"
    "746eV%HP@aFMdirak}7z@xc@5Pdv{3#p9`ePJo@U#7%lM_Fsa|#7oy~{{wQ-o_3ybr=)Qx_ql)<`z`c#x|VU<1;9_TkpA%sv=8U;"
    "d*+G!eLCaB;}~BW!+wU52ffMtl&6Wu;l$^QeD3}H4x9c)Y{=6dApY)Tyy~~a=Z$Cf-v6Afy_56&9rsH<VqEOe321lzBFszr73Sl<"
    "Mfg7Mln=4*`sq#h-fj2y@O@jSrTE_S?aSRcg|zqi_&#mQOngt=`Wu`pXnqC0w>)~N+y9UIf9Cf8qrNzQ8UAjFJ11jL?Sk)Cj(dHl"
    "I|uODELSEwPp&R^rza{my85(J*Sq?(;!Xcsn*Z*%q1@@Q%h3Mp!K>W$+0D>y)Zf;)vSR%_*YExJZTa6adhmbNy82_Ux%-nhulmOI"
    "bJPvfU42lx!nK>zYHGPVjhoDJ{XX-}5_jJpA9VL8Gm6~r;qU?Py62qo>{eGl>G2QT{X^b&Wjyocv9A7;gZGxZ{j5`6KhdL;+<0x-"
    "U{{ad+KTsmsrpM-Hht5L+x9tqxvNh-#`R}j{ro$3{{_oFbHD$&_ZqxcvsYaGnKKr+_M7Lpai+K2@u?g4-`^SS%I_bax^iBp#jf4h"
    "o87qWy?0D@{f{|$rR#6p0(ae<_Jw<X((b;wt~~yvx7~eJ$1ZjEtvk!LOGcjN-Y5P3hY8lP=qq>sU2nVf2zq}0q3d`2z=iI5>wmuI"
    "?i+L0w{G70oaV}O)=h3b;^$tT@7nhp;rdBGnc?c;x=-D>ldsOkxDR>6&0BKOb#C0j4P9S$W!Yf&p7!B?x^*`<UghQ|Xmi=p^8L;q"
    "*$1Wo2iL6ut`7eY`lNCV_;9lgz}*{nbw|AOT718`{8@bGylfS|hwi+hE#eJr-MKq+`doMJD7xxxoWmRQ3C<mj?2hlG7q7<m(A(xW"
    "!TlHE`}rqdsCDl>xzWi>_{R~=z+Yb5=H}r?7rFB?@seil{o@Cwxc4#bM(uF-)m-A{+m4;-*595r4)gO;=gqGDIpZAel9$#xxd~36"
    ";;tu0PTcM8FPk#O-B(xR?vHwIa5%B!igvm7*Bo5s${7!@cF*lIw#JnY9_`jO`1p?#T>097uU+{)$V+lhgS#*7cb;2cd#7=76^wgr"
    "72e}zxBkhK3tbsJ{gJybtlNx!Uzq3mv%Sx7<BA)feBC`aeEd~+eR?bR-r<0CPCukqFLv{he1GJ>-2Fd#5cqufM!fg6r#ZQbTHf#8"
    "KYaW~$jy}bz{mX)-TOrcU;Bou5B`~x>!_#(@_qO4x7;|2f9IYT^!S-upQuTd>py(ruTC$SCO>id!F27p1>^k9i*8&4n!5K0&w781"
    "EARX9a91AjF7R~p66nLrZ`tGSyY*aGCU?Jy-+y?=6YhCe-s$v3_=|p6#{plv--ENi2OKS1?dH?|V$Cyd9Djb}BX|8b=eqTbO4>nh"
    "l-z@Ls6Bc|`F`gK)9rtPeH-u$aL|Ww>Sr#4-J11x*ta1~Vc#~7M1F!*zv%1a_V)u1$G&4b@a3mQ<9q7b3-O(FuYoxKcEOjx!POTZ"
    "1pkR;)A?>Z=DPFTJln40+`6QvJ>v9qa79~(2m5l(K{yZ61@C{)mUZrZZb4~JJi)zhcuq&R?qUD8(BGWd&c4_cEN4IN<}1GI2K?Ug"
    "X7pn_xplQgH8`hs<5LbNNiFP`J^uo%&#b4$V%<78{6@`cu>Q;YyMBUECu5$wp5*j&uo(5E?M^4}_PsH{+xnYEx!-r+=H%C0e7+lB"
    "`nNwgJVy_Yp6%-Yeg5aJJZYetmpFb5IN06=_T$D*PR>mIZ4RgQk|!~rN1f#4BRv{r+@~6H`{W^Sy6Yp~dk45!==w>2S>on7>O9}g"
    "qj_%5OYZup(U_<E#{us}J7G5-|HT@N`=1U^(J3cA<=Xx9P}tj3Pjc^--UI!Ve$*d&>!@4YJVlq*jKsX3H_Y|B?1@FLEbp)Z_H=v#"
    "yKu|1?s|tgLt&2(`@%i9#ljJ;zdvq!&DFoU(CIh3tkmft)8h~~{%FL_Zk>W3js-91y&bqHe#hCX@aHGFd5fyXy630m7uNtE33%;L"
    "oQJh%Oa`x7G64K-7~)~kc`JaIt9!wo-1mbX*ta+a`2!Bd_ddz|D%gYdUm`Ee#<!8T;>zaj5J#^;9*WO<;QRTb`s4fhx>J!?V(f34"
    "VgKUgdiQ=ieuZ&<_Dk5+BWEsi?=!fmn;$!PnX~sv{Zyw9qRm^K+@|jy2EDet272$fPn<m1%0r!gjC<~JJR<z%neKY<&K&T8uTL5a"
    "eZAPto9Qsp$$ivu*)*)f59fo=mAP?+cbDvizCQ?d{L``EHGS4KK<>_ly}Eped!OLFJAmJVhB$eP>u+)Lk{s2z-POC??{FAj^h?P5"
    "i6?>wZeIS0>;Iyyu5Nw;`7t;A$hxw5<$UnER!+Xt8K;6zzEbDLAO7?sC$CZ0mz<p1+i=~CJjS&*-+u~r^VTmNFNq$w9r)Na#yv0n"
    ";86Eo!SSy;J)b;&(GuYJX|(%bLR0L6ZE*ND&)(<olAf|-7wqK4?tWWw_%`5n2JHWd=heBgwDla<{^u**dn7Neck7o<Zglufdo+xN"
    "-Jb#e_R#O$?@9ZcUVuHW06%TuIA?t~8us9Y8nkce>{hsQvXh(O=XZSrJ9N&+uH9vaJG+%kZ*-+O;{4Cb_dBnio_aR?n2LcY-}@Et"
    "k*uI!`3d~Q8>Ye!8QC2E>PMCEJ6hHPA7!_?GJW(8oR9v0rz4-mBU^xrU*r3Lc<x6n;U|wpee+jte(iP7Ieujp{0Mo8o@<AE5|>^L"
    "yHWI_le6@~5c+G(?_HVheAemv^ti2VeZpBa;Nzp7h5yj~6sNC}6OI92u)W+oMc4NM&#B%Fy|VUFH;?h>B`NNE+~Feq*-cJwnyqu)"
    "`kBf9J;~iSI&^-AJ?&@CK13h39S@xJaPuCntb(2Eb`bD=`j271?*6UoFTQG-<Aw1%n;@4*{nL#vd3rJQ@3_|BTW_2&4tC;6_nh$S"
    "x164c{^x$^yURlNzES^v&Tb~39s-_oU4OUErsip9$Lx+akh^xj1kd_l0Q|Al-#Px4+_HT!c+(1}f0JH)onDR~8}9nIGe<-2ue=BP"
    ";Gmzq3m#bmzWGEC@TK8D1+OiN!PD;D0N#I02psRcV<ULlYVf)pJ6*f@FEboZjPD-yzI*-|FXH`o4|jIfo<0QlUeed`GdnPW{rO_p"
    "m%zp6u>bA;1-*EEGuY$vFM!?ZJKK#joVF;yzU2AfQD2RRfAdHSjKAmGST}zEVI~@$4>%nB>K~JNUgZkJoBE7E+^2j5aMO8;^Fz#`"
    "S0Zm&@s)?+yz>-?kNDFcbiz5@!;v3s%PQpm8vg_2VHx*>*4PK*cb5IHbodBMu5o@r{6s78uHU}p@E>1)qVq?>zr)Wm1LnXUp7y5G"
    "yY^z};q=Sv;E&8(;MOA?GQ`PY@^YEer(tEOn@98cX!r}sjp*mAhhdK{ZG-jfvBT;2=&nxC-!;DgziDU#|7Y}DUqeoBck>-RcLI1-"
    "kL%%oK04RQZFF=0nXcdKIzaBuc-X>jb%36G<xTi6JLfuiFx?+<{ibjAhd+Ms7`JZbwz<yl4uA6|?AJkPmoys%x%#G!n}_7^4bBcm"
    "t)2jXdw2}w^sg5{KWw?d;lMt1gBzb+dJgpN>FpsObBn<1Pwx%=TG4wkcx~_3AkP&qfxirbUi#M{*pVm89sbOb;YhqkG7<K+61;VD"
    "CG_(jQsC;fz&$@69>bn@J_3H!BO?%J=`vyj?Cn)}UW-1Mm*!Ww_YJT81U&CAYk;R+4}1q4Rf7-QKGf-t_~kB6|Ac=n0?+(xHTcIZ"
    "gAkXgd>(qKRTb8MYPp-Y^v@P}WBe)}$o=zGj0>-sjQHH+&EQYG^AX~fou?z-H>(JK`Wa2&Z}+<w-}AiO4e_l1x(;%8+h*|gAD<69"
    "yy!v4-{X_J!0z6=b{q6(3cu}p$HTv>y$(F(_V2z$oa8#_h0fr$@!a<CPl~bb=CqyQn@vx1_k~wH20c4%l6(L3PcyK7WgGF{AKnH%"
    "c;}zt2mjx9v)%mfTC?wV<a=fLE{)%x#T)ruO+Ed&RqT)aHLhm4n(a96!FMdbV7Z0&t!KLiuFu!(cRR~<<i*?Ap4ZJLwwL#LpZ#p$"
    "?`v4+dqg|g565NXJ=Su(A9GxMPPoP&pXK*;X)W)6o%fApI~nK5a_{>i8E^cyKfY)Tf9G?8(Y(IJf4_J>`{i@9JU#TM<A3?r%>vf>"
    "elh1g{?ec4gzufmd#3y6XD|!#y<Yg~WbU($;5qb%xF2#Y?8&ebU}rwqj`;5Bk0C$cfbGZ!xM?(aR+A0LQ+EA2_#xe9!4En7X5_#7"
    ">_XU+a4zydF5HIi#V-3Cd@KBs<4fj@kKxZhd#B^aVf`hr1H(Ex90j*M;qVp>TmXJub|LUG{SUA!S9X9NzT#c*qlFj4PyXlIh(qnX"
    "!10mzh8o0o?>HIr@@OgS#g{dxPd*fKzu|fpFNj~e!SRi7_a^A0b4R<<o>>B(F|I%K?q3(cp1ii%$yL<+QN)3_%z+>Ak5cD<NB7>k"
    "5dKFg?B2%%AwR|c01x@;$xcp_Q!aAx<@CHq-22-h12L{2l!5=uX%GH%Y(LkJ8C~t>B^Wdhdh+F)VP|W$ECcVl!0Fet?^gK34{S%g"
    "cfl=g-c09Ch*#gg#q}FpUHuB;6OXO{fA5TW{I4Aj2kG<w1b@GIC-~;N!{Bc`_@vYG!Tc}apFF?N`BTBa#zBwYJ^?&<$$gFwhE0Bt"
    "_dV}a=T{~x=fM8-Ee1ZvE(MQxZZGuT-CY*Ku0IC;G`6R+cfsGfj7Hqy68F4##YxUyMZ^98`*rJXrys)AtsP$rE}4w@;wyiBu6+OR"
    "SD(jsvf&9=V889ggAlK}1biyEb~Wrr#g)LvVJ9GdH46IH?)ZP;S5Kej@L_*`2>kRT>Jhh^@GInFJ+C|R+jSd-eUH1=0uOgAguhXB"
    "EaJF_T?qfYV_$p^xn`@=KgrE^fIpQ@Lwx6<_uV{(6HWrpeEScsj0c>8cy4>dS>o`1@TvJnLmv;@3EuvX8^Bi<flnn*jK;pk2eocL"
    "!2D?x{O2F{g8a2S0eE?K7<kq66TthQXp6F}2jV!@9l_%|rr_5j{)Foh;%DZAS&)a)cbtAp|8SJUb$IDGtp6zncpH5--t*GMz~Nn|"
    "V?Sijeb|>6{UG*hF06BMY&U%$@u(4Bxc9OPx&s$4u7+JrcD#;#iqWvwcl1C!tMvx(zq;7TaXNJNQp8O=A+CJL&#@mf<#*7#mn^~j"
    "9r7*SYx-I6hyJm2Ht@1xF5*vPfd9F7yEv4&=`84%D=VB|m+Uzca{ZTX=<oJBoc!9Gk8!*--gpV(R5z|iJo%MHP9LUIt{w_Kw+H(q"
    "b<YfkAHNj%e!T+mj<W}1U*o>=P0%~nI=vTuwGPic>T393fBeSrxcKT)@Yx~vKexaANMG&+yV3nc*o&WD3_rcyT=?PZo1tCpRjBvA"
    "2KhB_IvD=>k*~uqpa0v=h}X78{>qz>uQ;hV&XwlfzoG0?2LJq}Z;>Z;^1aCO_>0%U$9~iRf4uGbCiw2*Cy1|{_Jxx-)2jyYqxHAK"
    "Pv5m3`nlPnb%@_Q?$*J)@+$nf?Nec=+t0`T!MqOO!N)y6!_9L`%u907iB7-A>*v89*VVwU{oc)puN*QF^W5~C6^MI|-vJ(*VxQ!s"
    "p@=J;yLdbNRQP$}MQgAh_~8oR;rs8qxUn(I5SOWb4}L($bFgpM`X$7B`fQv4ypD8b^6R0_-UUBKeA!ID6XX8vpAql6cQf|m+LVBg"
    "R<{B_J^WSdlMcVj;UK)NALOY2Ag8aRw_gN4?)fF`+Pwkht+5+$RCF->!25m&{y+b7=#hb!!2enQJ%@vM;xfb?ikFXs|AP08x~&I~"
    "x~M7qoK^$DZy(zL`|##c@Z|_R$}AlYyxlk){y=&e`0Ii-&MqX^^>;WA+f9O>H=`@`{+4hX;>RZf-#<JK@sd5iMqK!RhPm}ko4o}4"
    "__N?&u)8NiKmPGr#FP8nwut!XM;=*1KFzu*Bd;sv_wZ%@^`xKQJ}mRw>H8wNPqhnFf1&it`(=EF^@QULcs<~CQ=-Pf-;+{*JZY(a"
    "UD`#Z{&Q0GyF53$F3)3Wk7u1_qUtiv@Hqc^dMuxR4BJV`>y~xHGC9g`mmJOc;`NYadL(~mnf76M1fRz;J)F;B8M92Jozx{S6YHV+"
    "UGmcyX~#0r`OqXUhRV+;d7>mwSuFWmRXxj7rt)J*UKq(QBl(rGd^(b!O!6$Le0`G7Bku0Evq~;y{f(B_{=R%?IMO)j=gXB<a+W<u"
    ";Y#o!{ikX?0iU0$_C`G?R=APtBEON6Z_x|3#K%q6uGE)zX=hkBl=t{J|9T*Gk^f`;y2yRZG8MT$+Q&gA=Y84Fk$xGn3@HC$A0H1<"
    "A0_83Q_0U1`tlpAyiL+B<b6ivsWB?gq*3`ZB@a@T=SK3v^zh{|%NM2cT}s}UERT}P3nOwaaTTlc?;YlsI&WntaTv+xCwck0`0{F%"
    "+)5esD~uPBJCRed52@1oR^caB{U_=<B7c#x116_^s^nQePp*dwH?b;>!imO<wr5u1E5nZ}V~wNZeZ0iS7PcFyvJ2VotVinii1-Nv"
    "FMWNym_7<ueSEti@}FuPv5dqnh};(!##feKN$M)!nNfMdCBK-?Z<ghG5<4XM_Ea7vxnHhJJISvUi{0qv+o43~yA(Vf>f<O;dOXe1"
    "?^elaCg1&)o>u(GD0>q2FN~L1$(>blE4<Dqy>1kb3xyvYUzmSoa$DldtK28oGrceU9IxyI%T&p=QSu!qyO8$vc~IQfw-Z@i<o5_)"
    "o`u&~CC8Dn3nHI#UG%f?AEWZTW_e@P?@6!1b|cF_*3;)jk;+#rdBH?3vpi+peETBu9;-Z-I*+l)t;oH|xxOy>!6ctvoRdFK>_?*V"
    "w@RMXRQVgR($AsdJ(>R1@~H46IEZrmi7=-;fY+2NeXZ=5Eh+3Th+Y?5i2oq{M<vQ%P<BAdNc$JZsdfq{!k2_!iG2taPQ>33ycm{N"
    ">3^vw#78XU5x$=w{3=oS5q}}h;Zup=<8U88k>)!JH?hj=D*l7$Z^1*xlY034C{+1cv-UlGoQNF=RX$_kPf})i@74c_2vmMn!H4{x"
    "3CXJ`c_!s|$%AK<-WU0f6ra)lsr(+46_)!z;le6;*SOL4Ay(sxN($ST3<ugyD1TnsiCxk736)(5jxT&Y!^N?_J<<M!_}#JcyHmxF"
    "GQZ;}A18tj;YEp(f3YXxXItej=lR`IN}kL_;YO|-<!?*g$uQ^tOC&mPzw%3B<yU8Xt;n}OSsr}JZ!G!0W&c+48LR&nA^6DhyA~<?"
    "(Ou!Ji;oALpIGwV$7O~6>qPMutL%#Q&vl$C?N`|U4|3vEq2gmk-LG+_`IojELFvEu)8o>@{)5S}4^mpiuY^~H$NKUv{3;ZCag^_8"
    "Tje*T8VAZ=2(OBd^!)`LM-qNz6b`by;72I`S><z0R9@oj`r(CfBXOTl<>^k<|4R{^gew25RrWz&7hFibRf#JJZX#trLY0SA{$E6-"
    "{%?f%-@>cpyhN54xU;Xf!_vZbLwHgaKT>j>*^?aqAoI6N6pvER(Krlp#v}Sa((zk$U#RR*qT{zEh5Zt%cvYZqCHROGK5P!p5`37W"
    "d|U{xik07CRa{Db7r#5>PknrxWPZjGIrs^ce(&w`u0Z{tq(tQ@*Z<!r)B7qOrTrGM8<G0ILb1wQoc(_x6_3jD;46RIsQmXv$$g;y"
    "uS8}ybe??4`>p<8iufUMX<?oe_siKwP<lPXS6*DI)Q`&=?XM`m+>{o+uE&>@7LI2nsvo_tA^v!x_*bm>TcB`lk5&G=vLm75S>k^r"
    "f|sLxUZwd~*thULMz()(gwMNT#jm6ke?$0Es`e*hb^U<<U!?qy?EgOK|2x#%_rryE#mY~~;!Qo2Khx9qL&P3r`SDeJ$>{v|-F^Q&"
    "R(0`n^#8NV@_oy=RD4(Rb!Yy2S>gDV-oHyr3-3E*eu%O!nok9JJWKC4Dt{!>@m1wtM~YuXDsGiMr_|q{i1j{*vLAu+ON`pr6Z;{4"
    "h*kR<R_*6Ss?_nUxNl*<T>OuqPhmep?1fSLdLrMk!i(6EMD63HDvl*QDle{^sQ;g0RG$1m`72pmt-J5PWdDaw?cZg7MmOK?SpEOI"
    "dMF&J|JRlMe@EeE>i-Re>i=$Kzjs#hSz1^>hiV_-0RLOz$*6rg;b&IG!SsG$n6nSB_w8e44~+6VtiqQ($CefLYa;djMqLk8UHe!1"
    "{D9tfF*$J$8BbhR*dA(sPw&fG#YaQ6KC#j-nSWMRINqn@FM8kCs{N=y%|oQl9mML~gVp1c`&67W%8BR1IlMuik4V&eTFuW(3(Iek"
    "Gd?X3nZJ<3v(lXNB{pY2LdT1ZnpcyvKdJqGZRc!G{2<P;Z+ZJN`W%hsXSVF$+fAeTP5Kq)k0z&nfL-iYc;CpD7G4*dXQxXG+vT)h"
    ";r*dR;YP<bwH`^;eCarSTvB*FlhVSt((#&HxmWX%@fC$%tMr@gR~Wy#zeLAb%L>b_#;4xD*7{z*_W^OL99-q;gCHmV73Yk@YW>(R"
    "Xa7*y0jtla<k&C0U!Ur8X&V2fg~uD07Uq|_KRr)I?>lKbR$4e7spB77-(~utEJqI2yvAjP<tom>ca-xyZI^8xFVTEIN6+c~f-pz!"
    "^Ze{INB;hk{?hB2<mfkjZbQ#+o?g}Q+eF2`4&X1Pg?XLLIX9y3OLUy;Kb#*6bNGuX%Ndu72PTT|1v%qRa?bTcIlL>(!C}UGN(%dv"
    "VGbY4vr~EJHEm8DGF5yx%E3vTb3VwF6t?>Z#CJ+_`puKqvcmR8$1^j2R8rV)$mQEA4yE_W^*HqS^!i%m*T*^jf1HE6Og_}P;JB<X"
    "U&yl)nm=TEwxqCK(>&GcIGf@pdcQ{RFX?#30q4uqJZXCssp~q9uG{C?sUS!18I8A+!g?<%&Ea)gzvRSyf_{b9#pLWKq@{)Jh31D^"
    "AM5#u|3f@97ymiuJo0cEl@yKxYkwduE6g*KoO54c4&T=Af53V5eued&-jB)iw@q1Lz7mub=FN$c1HF#+Kk&?`EXQBU(Jz|k>vMFW"
    "=6%|plorNmlp}vh4!;Xa3-f!U&n1=pdq3oW?;vvQW!S&4d}@E+sP)kPYM{>nmKEk<djHSl_!lN;e?$ArdFR=~Jh@imigS3q-hb43"
    ">VW;lvcf#n9{=zCDm`C$_FCh_mgV@BC53sb)$!|`eI`9`nr9zyu2Vhdfce&TUg=x?{4_^S58$c&3&&|p&UtSA`~&QFzrwuV=J*SF"
    "aT)DznE!C@?*JTWze~qQ^LUNs_gc^9%u|r_eMjE+RZ)(9)qEh%*{3<cU+Q0&=jwK04qppObM!-5VZU3yZ<rIO({VtpM>SsazN3lC"
    "3d?DjbFNdzjdfhZ<c!}cd!y~W$+5>8$2!gwmld|lvC^k{{lc7c7h3K#&(MBMloJQe!&~0|UYH}_dLFcYs_m~W{SSV>j^~#Zw&!{C"
    "Y5M*9c!B2GNzVD@jOUjY))P^Vy|rb9<BujMP7~$CBcmMuGw-{COrA>$%Xh5f8k*nd;5p6VuX(&a%!zwxdz^j0nqxnLynV8~eTAI;"
    "0F$G~(!BFZ8s9nl6?yUcFh@?)ynU4Zh4Eq4{OI}DxX}JVnB)J)Ir>iT7g;sm`n$p~=Uio6T38OPT3;=<alikd2Xpo{P0l$|{T)rD"
    "aA0!cq^7iRy!?RuN;OYmPCQxr({X9xeKp}@VUC{`=IoPcy)N=}KpaWSZ`r@!Z?HN3P@2aR$_mGCb$lTEUPzrM%Fa2cbJhC1PpZC;"
    "N!5AfRPFm`=PUVt-V@17#e7Hd|KlUpQ|4(h@_*u)Cn;Lz=beg}mnq`!W`&;*Dqy~-fX|C1-xc#$h0M1XOCBaZ*M7>pN>VZplVM)0"
    "Nd7PV4Bp3fk@PG1r<k`X{DdXnC*X6F<s2XLUs=h6#pl>c_RIWS$!<Q6^JpZW)?}8Fmx|9zCi(LdNIot0V>v&u<geoWspNUy;^*gz"
    "m|qHc<o1=4FN=A@QjXWKpODXy^$S=x^1i8`cP!-dL_Q4jd4*i(^c$9{pP2c&k~hl>#?wNr=e3_D`c%%V$~kU1S10F9i4XbiI6J4K"
    "&f&{<DPh~llfOjdQRJF=yd)2p$VXbmc1!*BiY2cb^O?n)`Fj&y|IE+7marbNw3Pp78rv}+S!{jz6I{qTL?8G#lKf+`o++Pa*8A^c"
    "xL!u`l(C<X@*GfZBen}EPbu4(sVv!lK->lOzC0uo)brN(^AL-ij`QU$_{Psq7jeBrE=6A?%u5&Oymg{C_OjgL=buZN$1Pgz<2GH$"
    ">+HwY`SU1w?n3ILnB$H1`ty}i?}U=KPv-Fre|`d44=GCu^IxOS<tRRzox_v70dk&N&R@#+EOIW4-<?E!Z_D)ycd2}QlJ}JIoXC3f"
    "ebdE!Z^~E7`LWVZwXc6gk0oM9ByXYQ36%8}y<YFzlUV5;BYJX!uUE5m<2nYreElu*ZulOSa%oy|T$GD|a+$otalFUpZSnIA2FyQb"
    "XZ!g9MQ;XtFSE+m+cD?Y2u|2vw9MD@Vt)hb@uUUiVVAG(#a;%rDo^8lU#}!%)Vy;Y#cl<XkCJ-8a-4R9pFc8Se#VHuhjo4)#-!fQ"
    "_ZV>g%~!tMi`|H*SMA&WdS!YxCqAX)j)D4KLU>iI@&=^pyH2aV2gts^l=Gi*URijQ*cUm6&oU7E(%!!wiJfTe<1m#v`Iu?p_iOmO"
    "*qNB$Q$}nj`YWLPhJ0=;JWldkvRy2Ckn5I;o%&Acd*MBl$3*m-jDzc!RQo*0NS;m6-*Z*|%?-Z26a6f9(^4+&Ov=ss{`K%1KQE}6"
    "$5P}_cpAqO5YI`aug8<IJ|B$4Zcp&-lGvG0;UwPV+hf7Ij7#{Y@YA%x=auQ}e!ft#2f=%OUeQc{aGsE7eBb*eDo?A9KWo2L?fWO{"
    "`v>{X!Rq`o<X2f93-vvJs?Pn(_h;gF2=9t1&k5zt1U~;yL>~(OTCeieR{Qgs*<a!BHU4^9>KXA{M9xKDC7XTyCVEoVFXIowQ=0p7"
    "AEqkr>=b{Ugl`G{g!hTP5*`rF@a>uKuW+sJKL{S}Hf0wCKht@?;9Tq|<=#&8?M}*hN?4|!`1x?78sGm)gcq&!^XnSX-?YEsMwQ=}"
    "yej#S<$6CaZY=g>ygv_;UpSrU^UP5E7`|6bI~Y_cUM09%S6(o_kbTFY<6R0L`g<Vx{ztwG4b*pS`n&#Ao!gi1@Z}tSNPQmDZY0$A"
    "hVo{bD7`N0!uhe{M{n}yO?XeN@-t`kROw}|L%2}!0m*ADyo&RjP#(p968$TFr^Exo1u9Ok+1KM4KiTc`c=7WkzqQC;@RiTIE%kCL"
    "d}t0!;>u3-`KkE*3EwxF?)$xg_z|?v2l&epXOQ>^@em0PSNMD<T<!CujPH;2`KS0Dah>ANvc6k={gSR$<KFAfi{Mc7oUDuZW5P2t"
    "-kx)wT*pI$euZ(OzZ;P6Ok<U&BUay;%6HfD-JDV1lgjsT`nyuGPlk2LKVWk55J;YnRDBnheW$APcnD5A`*DMaJWcdsmY-etMk_yl"
    "Ccg_midOjcPH-vpNco%D`lvWS`k61knLe-g?N>O>_uJ$3zTL{~8}*D`UD(eOeGqO^_DaPGqFuh<94uFICwbG!bJERzo_U!M$?q=j"
    "BiF^BjMn&eQQ{BMzZITHeqeU_d@$Ok;yfSwyf|6q$FuZ4u*u28lj^v4&UbWaP8=)Nc`d|$mpn--@1yPL$93c8ekt*#RP-g+S>h&Q"
    "=gnrHH;aEPav(fk^jEOnm**hx<7uL=MGl0IWW0H!Z?A=iWqu;Z6RUjx!t(;kf57()X8H0WJSmIMuJYrn=@)+gNtGX;kT{OSGeoY0"
    "=ZM_Oc#;J^Kb1H@7Jpdl<5}b^oaoO(BL0l<{|}XawcC%Ui~W}Ml(?b90c76O6~28ExzF&kM?LpD6~~ge;(Pw@5*ImO-z4X}V4(KP"
    "jLKJ#s{9;To+g!VDa&6We!9v7BYt`!ezf9Ovd<xWS?s;&FOmO@Cv5lqxGbJ3c3=E<g&V8=w_QHpN)Ph=5{Xl0JecxqB;K~fUnk+Y"
    "GF};f=I1U{JbA0~!)ZT^@Fj_Vkw<0rj&>wzQ1Y?DA7?<khnsyrKH1{)cZm;3oJ#aX$o3MyNVoa_Pec5iWU{iGyL_H0yf~BJm1^8u"
    ")qd4CzP=a#E+w8);d>K&e^2ycxYD=7q92Xu3vD;@@+YV~w2|gfeSQB!_>SNr`)*k0gOu;$#eT^5!1OB;mLiXW3$X{G__ejZ{~><7"
    "=t0@{Fyg-n-#^G-Pto&P{80G&3dJkPrxLN#^?p2B;)B`$d)egsCE0$8;QU)Z&Mk59MDYfZPy3yZhh&QHf6M)m#9_q$7Cj+6XqM0S"
    "Wgpbc_Tw2+1}psiy>OntkC@qIiBn8c@um$b?n^so)c$5%>GJ?tzrYrbFAENZFJ}8s^ZfM@zG@|&_HMa1FY&U>9{+!Hw;nD"
)


def load_embedded_data() -> np.ndarray:
    """Decode and verify the embedded loss trajectories."""

    compressed = base64.b85decode(_DATA_B85.encode("ascii"))
    raw = zlib.decompress(compressed)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _DATA_SHA256:
        raise RuntimeError("Embedded data checksum verification failed.")

    expected_bytes = int(np.prod(_DATA_SHAPE)) * _DATA_DTYPE.itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"Embedded data has {len(raw)} bytes; expected {expected_bytes}."
        )

    data = np.frombuffer(raw, dtype=_DATA_DTYPE).reshape(_DATA_SHAPE).copy()
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Embedded data contains non-finite values.")
    return data


def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
    """Apply y[t] = alpha*x[t] + (1-alpha)*y[t-1]."""

    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = (
            alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
        )
    return smoothed


def smooth_runs(runs: np.ndarray, alpha: float) -> np.ndarray:
    return np.stack(
        [exponential_moving_average(run, alpha) for run in runs]
    )


def summarize_runs(
    runs: np.ndarray,
    uncertainty: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(runs, axis=0)
    spread = np.std(runs, axis=0, ddof=1)
    if uncertainty == "sem":
        spread = spread / np.sqrt(runs.shape[0])
    return mean, spread


def configure_plot_style() -> None:
    """Use the same plotting settings as plot.py."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "lines.linewidth": 2.5,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def plot_metric(
    data: np.ndarray,
    metric_index: int,
    output_path: Path,
    ylabel: str,
    ema_alpha: float,
    uncertainty: str,
    error_every: int,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)

    for method in METHODS:
        raw_runs = data[metric_index, METHOD_SLICES[method]]
        runs = smooth_runs(raw_runs, ema_alpha)
        mean, error = summarize_runs(runs, uncertainty)
        iterations = np.arange(1, len(mean) + 1)
        color = COLORS[method]
        marker = MARKERS[method]

        axis.plot(
            iterations,
            mean,
            color=color,
            marker=marker,
            markevery=error_every,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=method,
            zorder=3,
        )
        axis.errorbar(
            iterations,
            mean,
            yerr=error,
            fmt="none",
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
            capthick=1.1,
            errorevery=error_every,
            alpha=0.9,
            zorder=2,
        )

    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, 100)
    axis.legend(frameon=True, framealpha=0.95)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot embedded BOME, Sequential QCQP, and PROBE mean losses."
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA weight for the newest point; 1.0 disables smoothing (default: 0.2).",
    )
    parser.add_argument(
        "--uncertainty",
        choices=("sem", "std"),
        default="sem",
        help="Error-bar statistic: standard error (sem) or standard deviation (std).",
    )
    parser.add_argument(
        "--error-every",
        type=int,
        default=10,
        help="Draw a marker and error bar every N iterations (default: 10).",
    )
    args = parser.parse_args()

    if not 0.0 < args.ema_alpha <= 1.0:
        parser.error("--ema-alpha must be in the interval (0, 1].")
    if args.error_every < 1:
        parser.error("--error-every must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    configure_plot_style()
    data = load_embedded_data()
    output_dir = Path(__file__).resolve().parent
    output_paths = (
        output_dir / "train_loss_bome_qcqp_probe.pdf",
        output_dir / "validation_loss_bome_qcqp_probe.pdf",
    )

    plot_metric(
        data=data,
        metric_index=TRAIN_LOSS_INDEX,
        output_path=output_paths[0],
        ylabel="Lower-level Loss",
        ema_alpha=args.ema_alpha,
        uncertainty=args.uncertainty,
        error_every=args.error_every,
    )
    plot_metric(
        data=data,
        metric_index=VALIDATION_LOSS_INDEX,
        output_path=output_paths[1],
        ylabel="Upper-level Loss",
        ema_alpha=args.ema_alpha,
        uncertainty=args.uncertainty,
        error_every=args.error_every,
    )

    print("Plots generated successfully:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
