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
    "Sequential QCQP": slice(10, 13),
    "PROBE": slice(13, 23),
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

# Layout: [metric (train, validation), run (10 BOME + 3 QCQP + 10 PROBE),
#          iteration].
_DATA_SHAPE = (2, 23, 100)
_DATA_DTYPE = np.dtype("<f8")
_DATA_SHA256 = "674a850bfbb1318921534ed68982a3416dc662eac76963db49df2abbc88227f2"
_DATA_B85 = (
    "c-nnkcbrzm)xIFT7m>EGg9tX3ML#S-ac^RbQDamTP4tt!YK#$kH&J1%F;Qc$QT762Luq1Hc0jrmsnT~qPywYbpzN~yk@K8$Kj!$p"
    "dH<Q;%$=D#cji2&&766oD2i7WnmF43d#1b#rI(f${wMWEBkvBX=t|`)x>C8FF@*AF(i(c6ygdo|(I}6`459M)-!+u?<JS~W{_!=`"
    "-o3Pj%9GRZUi0=Ks-M1vb~YVE&)w^jkRMk=?bBC=QMn!2742U{|K_P-)NlMU#t}U`i1O(>rv&{*BThlP*D!9WC;iKFsqZEZD>!g`"
    "^4xrM3Y91CPAWLCz58Gg`kC6b%Yok=eOl9*`u(^o`kUUB@@DCzj?|AEMtM7XQYWgncn!+u4xzl+Hl#iE^Ua_Rl;1b#5X$d7rG4<c"
    "dF{cJZ(m5MYg5sV`fX7``LsiATPi=akkY7A1*OT+rERFZxVAN=Cs(wh`p1@2J~^=@D34DmM|x6iD_+-@^!%}!mb@-4czuKV``4`n"
    "uWNIj$7a0VEqLCW^L#X=c5e=A&htw3%%YmcJa3JIdhZWuc3}VM+DVOqbnzOhA8)E@$n!>NI-{lurCWwk-p?CEd0VHpKJ{O(oYMU*"
    "Y2D*}B}l7)2eY>v^t5+q7v#&jhUw5CAMGtEL|VvcY2ktO;=SDe-U{GqZ%LN}%m2T>ilCnVse=0(3cBB00y^7U+coT`qRWB(*u6u6"
    "%e}Sy9C){TYq@<Xzo(GvR}>uBU%IyxxZGRF<%L{cOZ4pa7J|;JN{F8Q-ddvPsH&th@<T!2RYN->Ed||I6?Ql<wN*pgBVWsDDV3*H"
    "g_L(yrNBcq=-gM85#5_=;2^FpZHshh+XM4Sbs5opT3y;2=}<~-^-wDJ)kBGHT{Y<5SC_Wpd2PY#-W=tHtqx3GHRwI9u0Vf#%bFe7"
    "-~ROnUiOwW<#}lm^uu|#cWBcvEp5v4)foA5&X+gld2f2)`&?CdL(pp>=)bDA5zl8sv@dJQ^V*Q-yAj$IHo!bp9EAF1cwSwIe06DE"
    "%<s^8l;2)PbeF6{8m|I9nq{C9w-9t><{-7xnXYD(;rVpXT|5=^YG;;#E@zj6FMbZ(*eNWRrUP$&D$_BSOFcJpDDzpy5%+JV;l1fp"
    "#^+Sv$WN;co+mR(a_6}}d4Dn$_)n(+A7(~rmjnBgcIgb@BI56hW&t<JoHC++JF77GUN?Iv@G-ZH<pNT9?&nqjC-cf!P80%H3raeG"
    "j{!&f=bi9j7czd94h1fj0w3uT$cOkV$c<=4S=$3szqADMV_7MlF9)6^Z&m>(k|wJ`?{Nk2lB@x)%xcJ${p&<@AALTQ(s)WmkdJ0o"
    "G&``~{`G~tm<77tzivT(|9rOK`D)7RPVLj_WsooPfQ$X}LwJh60N%`;a>_?v08f%;?~(TNOM>?4qKXDQufW3+mJ^(sCBRLVt|~tW"
    "@*{$N6&b95l&CcAqRKh%x~jC0<xyeSpHq3y?G2~C>w$9Yfcl(9UGwB+99`6N&RZFGLHPc-GuPvC+bL}4__>wmox{{{{g_kNQKcO?"
    "ZO`>2J%r!GX?if%Ysd4z`IPgnEw|&;@^jOM%Q^L}_<8HF+;VxsX{_?{+_d25k~ZUUw&Zbe-Zkg;TrX`F*7r@hKdu*X-ZkO(aT;-b"
    ")0q3?e8TlpuIC$ad+DbE_apU0|EVA5qlRJL*5i7OxSZ+UNm`fdMLZ5l)7tV#(X*m&p<7LlUg0Lzxabnndn&k6II)Ex-cr@SF9`dI"
    "3PU^yJ!j<ws=P}$et9nM6MT4u11Xo(DSSxY3$8fzjE{uzVH94X_F-xmC$?RPA9qNI55a}x6P0E-VSGfKcY=r3A$|lGp6TBTE;w(b"
    "ozQu!Fctbw6b>YBn};|MJPFP$<06(+;V2P2H42|g-Zu#G<hUOvc&QiS#@Ek{3tNZ#;rdqL#YCZ85IJCl&vpsv+9^6Uif^Z?ol!V5"
    "dHB9gGr6VcJ;RT}U!=;t>c<p>@*q~YNEKcL2lD<z@}0wcCU-iAI1rpjnu@#-d~iLR2Oo*ZiGxG=k>R6FSf0xBHVRLza?2OjGQ>%&"
    "@zX-dpB7<xs&L>%9x%N|O+r2&H&J+MtmwN@Zn+}--Zl(zBX~#}gm@8oBY7h@X{h9k!cW#uy$~0+wj`jlIFJ4!^p-082t8*wQmHWt"
    "4+XjTeyr$S<1NEOLGFH>Y9Hsplj*GFlG5|Eyl|aDT!@?y{Yr2mdX;7R_e!ra?Zf8@<G>3pIE~sVT(k-4-naSh@*-*-$_wH5Y3oof"
    "Na~qh(-xsT5&VcACVG`T7ky0d5jPKUVVZ<`k(7)6mFZzhUU<P(vk*6NgYdb~v&f5x>lwiZ%K=|E#DmC(Sji8=^&IEZx}n^0THh%Q"
    "`L9)aidXtls_>%il0?Z5v3H_8_Ka8kX?nJ*+^g|rc7xK_BHd2m#VdRW{Y$;HW2jff3ODjxQn6127os<b+z>lM^eo>tv_q_t4_@h2"
    "f{*y%FrVS0by%M1TZ;Zo%h2A)<VuT>?~8sUc8u_Qr{shYdqm_-<Iqlt6&|GAX}yZ`X~R%H2tFdEuURDrq6VS-$n>bXx#dqLR|FqX"
    "$d852Tps&6Qg*b^xl`quZ(D_@Sm7d3?{Qk5by0SU##<g7MM|z{ym%%51dpl4kKjY=Qwk^YJXZF);3ZP_iMHFtPEW)>7d=ba?~&lA"
    "O(<_N{Yc7Nhx(XN^2R9rO7dx|kiG?Hk+LJiUXgl<qI1EG;3aAv;vsDs;zZ<%*d<=!L*z-)Q0ZX}LOCSm;tz02eq{RCL7}`b^+SA!"
    "o)sy1k=f_Up0%1jyX5vqIfWCW?R;f_TV;3Kg52%2A1$vuQFcaVM`*i7+w1E6UfVxPpA!4RC>+Rh(;>7=(hi~A5W7R{5|IOf3&BI8"
    "{a7mRS}Xgbbx7x#-lhFk>Uky?S}1%q3+0aJWf?t-J<>A7LDD$HPps@|u|qtkrcr3eh<zgVhgbSmr0fcj6C!_1gYdcZFL)7sO!TVE"
    "-U#G}uf={sTFZX_THaTv!M<Hm!+w9x$NTU+t;W8DuPQI#{Y8=|_9yHs_#b({Z4c_DyRcvAD|tU}2liR)cDy&*hJ6XU1^W=uW;}Nr"
    "(O>*6+DG4D9DW`43(Z>I&-7>?uf~3cUx75;gmL-B=*ND^`!Ms-kDZTk+I3}+htu#LHv{`H>16cd%CX-QeTera@AG#~<o%a1*cbI<"
    "cwgu>?B^sSxqLYGWBkh)uN%huDo^8m@e4eTC(xgNl=o{a??>H-{USGn_o41YKlWzcH=2m^2=Vw*N|S5RJ{^tka+mS@FU5De3wR%Q"
    "AoA&@*!N6N<L~d!@BIy*t0+OeD8~9E$MZZNgEZ;I^l=oAy94&qtiku#`!TMlGw(lj#r(R1F%P!p6uhq<*WWe?<J&ci=@aJ>+|FV6"
    "uAMliV78+?S&nlV{#!n;u^8tt(j_>Dkz5QqaP#>*#OFAN;V0vH^gilGpW)m^GI~&;_qbZ7C&ouyi*@!jypL7`9L1csjFV^|@BdZv"
    "JnzFgIo`*Pc%Rz+$o*8|x!=w6unYK0ch+JYJ9+<cJL<cw-0la=ck~1AgKt84v<Y~NzU6is80Xut9<Bm(5U*jnT7&&Ovz+n2l=mGM"
    "GhNT)ebNQI-m~z1eg@-d8s48w;_-aKIQ@`uFoE~^m-6|UF}zPX3g2g5<M)o_?-<7TdXee&Y35f?aQ+caA7c7^82rfJh51cCz<ZKw"
    "n4esYao8*PJi}i=&*{aC%Yl3@;#|h}Y4{HR3a?uU%FVGDZ`_CRdnCRq>cO}_jMuj(=sj(L_os~*SM^cf?Zf#W6Y;+8ZJ^U+56<h@"
    "oi*TV-!r{#KpHOrUXs;J-*fr=&|;kbF&_;<{{yhj_A{ItipK&^@kHP#N`V{q658A6aSkdO2Am{Aan2}y1nu1eNbNO)0=>%Bg0Bib"
    "62^;R`QmG^FK%nFo^Btn!#>`3u4X!@X1cH9`OnUapuhMh%xfZeVfhm8#yTXs@IG6~^t6-P?O^)d#{6ju<8%w3OWDlyx{=rQJM@!m"
    "z<0%8W8M5(jN4R{2e|R8_#Dj&-nU=M<6Fe^vZ$2t!{-TRb3Zfre2wg{Phnhr%5?NG^M`jCH{-d#@l3aG30}B<8PDs>JWtOvT|L9+"
    "a2{tq^bpsZ#Qf$S9`79(hq;-@H<<bOU>?Vn+|QrEx80RIzXO=xoX2=Qo!7k|+NZw-zxOBNd3q$v^CG5)E*O7&1kUfo-C2KV!?<q1"
    "=PQ1KTrj(szMEmb<1L`ybOX-SCEHj&tY>_z0^OP!Ot+5H$-sl33BGT~qP-gj{6#P0J!T~5N1#9V7~|$4mOD4qU_5ssAKfYP1AbU{"
    "C&p>NV?2EV{Kk)ejC$uRg+A`T!u;H~6z3;S_y}@&)(toh@#FD0|IlUy&I64bM0%|6dJfJnOqq%EW=+;Z?`&`x&f^?DjP%2#&ljX0"
    "#y#MlGgm)Kc9c7@9riVkdUH4FiC@g7G~IpN&JxnD%fF}mv=cW`ep=xV|4Hp1dT%ZvJY0hMXAWIU<#X$!-spd9ptN+;0_ykv`?mfk"
    "jUW8qZz=!88`M61`-=6{PtWV7Q+|kFP4(tAo?1fQ+H^Mc`~3Rl)b5T4seL-5fS#M-{okecobbXn%J+TpLu&u%2b3l=M@^voZw}g9"
    "Lj3MT>L-4D5{=gt52bwa&Nh7SD?1laTJLQdx9fh=D#}kig8H+s&-sqppS$u4dj7Yr8}MEAU!(l<CoiV@TjtR?{Y|&eqH+Io{1{4q"
    "akQs&L7Qb%Z^Df<Zg=nPld1o)$E~6MCN8FO+wV(yf7<-M`IH{rcO12=IC2HG+jtt)OGoyj@A1F*D8)J!d`0c=8b|99b)NGP_51$7"
    "B~;$zx%a5u*t@==dF$SfQa}3!T94$#Q46Vl?-A6G|9l4J&Bj?Y?)0lOG47U+(Y&Pt2Gh8s>)MZ^w0J0er+ef%T6cT>6*NCl(@R#A"
    "9AKZi2c`fAgEs(A-97?7%QxU$QT@%p^;>t5{h6-53g>%Py?}FYE7#(@T-(*nfcIwPm$4_!C%ZJhVjTQ5V?T#oI<h0q(Jfnt^K#o4"
    ")<ydPIA8hnzpLoGC)X0am^((0o#kG5V>`{msRPKKN|x8B?@u0>Lf>PXkKRS?Dleq@b`xgO`n%I6Vt!s}yM^kXHIZ<azVa>6O?1)}"
    "Do+pl^hau6JY@>CtFEN>ap%p16F0G757qzkK^2tFcw`;DxBG-jN*_F&);0R<j*lsQqyN{G9uIm+@2R17zW3R*zV1#-bQMi}V=ca8"
    "6s>>y^b$&=XFs8Krg{td{r3Xu&viYS#+B4Q{U*KFJo!46Pj5`$ZTfE{{@|}%M)Q&WV&tpT{?rG7&quDucVE?y=qhe>KYhP>@_NwC"
    "l!d^@{h!kJ#RpyW7UhSYL3AA#RD!<m9{vuEqws2aU)1RgTA#RX1@&*9dYJf<t@|6|54L^htr+K@UZ!#NuSefuPWxairT6`MIHiZa"
    "3p^dZ9Q<(5O+Qh)n}0`XdiUFS{;S)cqW4{XC-Dn&es8Q}|F7wJbmsBE(aLo+pYHq(&(k>m_SPp<{-@v3dd9t)gKzY@2kTIE_^y%z"
    "{7rtxUm<7vKMx#q=W~6}Ujn%}`=5}1kJf{{-7*sASJ$53gXs2}f!(OypPNGudu9yI+f|*1b97w>!as8EBH-Z43&?L^M{`<lA&til"
    "KAYy*wVz1q;`=>Dd^)<U8R5Z=s;mQhp&h>eo~;|{du~GNjy;CH*PPXw*4^}Z2mQ^POY+67=5+3pG+)VG*Wr1i8_|zzL+k1aD&fbt"
    "{u#naS_S##&N&zBGy9ndShrS$-?)Ay)_+wW>L(g~9Ok)wU*glzGUU@{yNTZ2dt-sOO*f3D=Re*;^lL9ThsNjsaVz0Deq_vC%Kzb<"
    "Ih6JtNb`~;PXGry+ChF?--hVS?z@F>>Mnd5^Vzd6(T6`AY0`Zs==SNBZ&Ufm_r?PkOQ;`zQ7@Y3xa~rkNBiQuSEzjS7|hfC6M^@F"
    "-H;nkp1%R(eva@IpV0Ris(1RqkhdrHrSJ0hfIs<9`haisyou&1zPNHE=Kbt4>UZT+ODQdBwHfmC{S<QHrWdHZ)x2Sl$A^4L?`^nb"
    "1od~v_BSa1)p^9<+{z;2L$*^(8h<?EMp~!nR}-KYblm}56pkl(Wq#k6<}I$6K=1b@7gPctDeTB$@Rz!iCqu7V-XD5u8T7jN?A5@_"
    "m0ciD?)zmY*bPVEeCa{h4^9_WKpt#bgt&q4$KhP_<qeuckFCTx<vE>jzOrW@oWHC-5$BF4{HZ?dhf(|J`*!^i<NV?x$koGUuB7i7"
    "T94+(4P8m{KHWE!_(8m78_}(Q_Ym;4Rh8g-M}1E8;K~ms{+M+BiS!6_(J53Ojh_d7;Oo8<z^|9lyxCSGiSFapE2m){ezg$#TrrKy"
    "+}&$0_<bG7@mUj~*L2@l1G+mC@@mkd^gYpgw*$X*9wmB9_T5DElJ>0KLHTy~6AqIB7lGc7Jr;W4mQ|lq{{yyB-u?#kW3M~aQQDw<"
    "A@sV&MBjeKiO?rstETar(?214joZIMbn0$FxgB`~)wjQR26FS}MWmO+58MWP>={e%^B*2e-xVGGI`R4R$pOoO<7ZLt!;kC14&6-n"
    "wlCaAc=0Ff+5<Ux0kwCf-L?a_Ga&zuJ-eFHq9*gG{_oe&ccd?GqV@BWYYD%;Q_Td({Ta~T9=e8}r!8*yH{@|C^wSz1=j`vsKpy<N"
    "67?IA+%mf-6Wv6=ziU0@&{>~Ry-T{0+)Ag{Qfd!9=Zlg9_A~s6XTn}79f<V3KLQ`=YL0jM9R7goros*x*#Lf^Ps(9;G^zqVif^XW"
    "KYly>=l^pO&O1N06}Y$*`$x(APa46`Isy4DU(x)!!7q}2Wfz}{xRn>1<Gk_3S3+(Syi9cF&okg(W3Qpq?|y;!y+3Ljt&f>q34MI@"
    "3$PzLo<RI6J?04L3$6>zQ+#cA=s7#LfUkUeG0kH#r<X^&CkYq+j2nn=+HLb`{p{pFOrmyU4A~v7-x(wy;*XlW51e$Oc{gh+Am=*N"
    "0lrWAHRRXbf1&=8D^`+Tn2i4cblLMc8ejVCGVtGtO`vbRb<9M_iKpp3=IeKePsIOwKlt4xhQ2rM)0^aGI;$o0oWXr)oo(f_B*)yY"
    "rl7m#7eUYZus`gvb>ETxmfo~u8T6*r#DCK+J%}$SPYkF2-OMqd`^)bEKd5uYyU-&mp>ICb3Hs9T)1lWEB+%3D-3-0|2m>7NzWsaX"
    "Y3rcZ?b=QClE2R&Ju$hv>;roL$uHyke;iJ7)}8by@V&eT>1S?W3i<P8*&^U#4&;CHSHTyrtq*y8&bg3VJ?7Fl&9tQv`~v4dkNWC;"
    "*f)<g#P~ap!@BYQp#2p4Z1(DI(69bA8MyfJGWZX=kAUB(WCU>2b_&@c_TbAA*H(D>A@GM!A$%mWe%S_o+HQy++qxF<VekJE=kzE3"
    "l5k-@;Bz>AE+>4LUVkRLAbF}W^sc|WL-<dwJ(lbd^H11Ww*NfH!+viQ-*p#&5Bo)fVUH|WOzUADeU#`h9aT*H)RY&|JlZ*9U@xTC"
    "qo1!HfjqjnDb};oF5>U;U2VX>E6<01Q_~do&zN_<2A$qU^Buo<4D_l_*TVjMd_K`_d}E)P)bHR{pu3YFagbZBz~^3j8}`fY`9u%4"
    "<73pXf2R-Z@q@<Fy4hRilih9p^fu&I9n?$fj|N?>Z%Xr!cH2yHFmC)5^tVUGf=(Yk7yM!Cb%X==%yl$Ax8f}D-IH2?KIRuduRp0P"
    "_-kp`WzcK8z5#kJeFgf<5b&i}hd_=zT|)S?hnbQ1j`UN=+j8iwTgt(oZ}q^{8xg%f8J<9%w>=bg)MF#yXK6QL1mx`%cwfWrn3o1u"
    "(D$0FK8K$7_YJ_)o(H}Ij&?#HxNR8mk7QIk;y>o$0_d4vtb_h>(-8P&%3lIsYFvT!pISom=Ktn^H}+p81KE$b7XB@}VKVl|o~#dh"
    "V*DrY+qIpJeVW+?u+vYj3wyiwz1Z&^)dBvj-wp=d-LeIG`>)S|9A5e$>F>#L?I3sW{dPO}r-$8k{L!$lss=+(x$V1e;3pXjzR(tW"
    "Z8E<F?2|&QyY06d`ewa;)XrS?1o-T<N%Z~xuQRZI#oyz*Ke`2c@Xo)%4*vghRUN`}Puae19d-VW&nG0`^S<jow&&Jzf2_YIJ2_p)"
    "^?2T+?>POE)2-ZY6W6QZ_4%6n-NETb){D1ueJ-~@aDDln54fMr{CorF`8(p>+z*e-%6ELr<Nb`s#qTj2!tpsif8ndR{hQow0@ssq"
    "jw}iGr)9j!xNv;&Sbom$iN<jG*6{n1h1@T{H;dz7dpdbFEVqj}&)=8uyeEsoc~1GeQ~6GR;Q3v<1m`)-tjWlqH3ItuTOZ<mkgFh1"
    "%8r4Y`EUpPyC*$?_|g755HI?}80cAbHzSVi+KsS7I?RS0((OjXM}Bc0<cXP&IN>GRah~9kIncMvsiZI2lRtw!|H7T5ADev_LJpL*"
    "CLBe#JVkhm2QG$wU3?z!GW}M_mCIX!4`22!^rIygz)pT{9Q;tb&n0~%xvmoa-P@1DygXh6d9kPx`N;=^?l)gc{(|I<>qy@)KmGuI"
    "^t&;Xx>I^V&zRT;eD~qSkSA|!A-akiJPtqb)_Jfa{#8WwcYN>7OJIK#LGFDv5cE^{FX$nsA4hbWo-ly?%l_=g==<HH12L{&7DNA;"
    "*8=*}k-e!OJ7y=%OEhEw_~fV?A!jSMu7uuoF7a32V;k(@2X?^UyZ9!WH`}%i{MGkwrGBF;cfJPy#N(@>zqiFa{&pAPz`yhy^!FQg"
    "L*LwZ2<(jqpC&#ZE&LMp$xBPfo{C<b2tIz>$Iye9-$(kOse28+_v{nNu1wd=gZ$}H2z*Xh0X^cyz2JX$w_5_a{si==37tvaMgM3w"
    "27ZSN>3zxSz9g^WvRfg)ZvK(@gIU*v^tI^1$?z|}_V9}(2cB<w3FoNHF_&SV>iRnHS6v8wD!OVN<VWe{z{erSz<)Iw{MPOIALv)l"
    "_9J|_-?xOFe&{~<tv>!E;`Pt&i1^12qp|OC*SEmK?Mq;9R2&IE?jh&FK5yLv=Vdl*BmR@#csul`;%V^jJoEw0qxrZm^vrR$QkwKX"
    "0sh<;@UtZ5e&|yR4+lRk+YP<_U)MojSqgnBeQFH$H9oAO{Q&#d(XgL?-39d5=osMTg)-<>(~p7P|5P)i#hu{C+1VO;Tx$>gdc<E*"
    "9>afTKb#GEC>l@v&EMLSaBVK0i1k0g0&io^#CKl23^=^&B<zO_xexmiV;;nQ&3V;C$L@z;z#lc@OZqOixFc}!@;b=Xbl02Mrx*iy"
    "eS0VPvzly%{#Tt49s6N(SHN%D27cw1zsG*el&isaFI<lKYxxbnYx-%hhyJx<F7UE>KKxH(f&ck;lOM|7a2oi_<)viTr9Yhly8e3y"
    "^mp6sM8EFFBS<ezzP}KDs_QqwpZwZV;)j09mBYa2e!@OU_4C7F$FBgs-z<f{<II8B*SN3b2k@P%i0_%NHsZZKuY~<|$9mG^k}Hd#"
    "&pvwpiwDXNKdKAlM#q;SFHXMzc6#&qu){aiN4=^mknj3u#9!ZV5bX2A-h^Gg@GouQuWf?(<{NOIMOu0krS{!_AnjfZ`}~z}5Jx}x"
    "Uc>>P|0eXYQ)^(4H`_$=!;Sh4{44#wBzm)5D&arcbTjPqJ)6Lv>o461|IL%M4)(RzVbATD3OU_kA@&azw1OUd)JrpHo*Q9a(gDX3"
    "e@`|ofIP0Qgk5|5jqtCu{1o$CZ~bccJ>TC2J=SBNr0+2Jm43Hu2kcbXd1k-{><50d8hH4{2jn-lb|w5WJKuvH(E2Rw+ckLw{+{mN"
    "e+;~iq%^&B7|FZn*YGdf>33q>fB75yUH5Lmeq7UD&_{PRhJM=Zb?lQ4zl(5SZs`p=>NAA+bv*86;NzZ)AlL4VFmJUTfTMzgU<clJ"
    "HT3_5bHGOiUI_bV)A59Z<kOY#I~1-O3Ht@#8+X_QJ!(Kb*g1^{Lce`tGvveDE1)mO(4*{%;lSJV!(k8jOQ63l-av989o&a-ZkkVm"
    "oj0RB`2JS29sc8Ef$v`(1%Js;m%=anzh$((zWytak7q=$Lhc?1{&>e#@F#b_WhvvMH|vqTSfA#+EoQy0h|fzFhvjMSu)Zk{>*;o}"
    "v{UsW)t`}mxxI|faz5p8MqD0ox$ULK!Ozp8a6G;!ESGw5QTQIOp38f)a(N%8?r6?)nyS2v(;O9+`y=`NM{qq!x!iHya+>xG>!pYD"
    "d~vzq)E~yrIrZH+J(S<asqe<`;WXhim3oqwxE;<L^<3f?tO?^EQXMZR=cZNsro<6R99z;!#p85Ract`Rw8RZa{DH*ZWbqafzjjC%"
    "mm~3<a^5@X7}j%&E@l0#rq@0peP=k*IO!eIl~Z(<y+`3n@FD$sHJ*sy?^S)P-jgWYNV(8&tms?#g6kFHCaYHz(!12NoVQHx$x&f>"
    "Bzd9#Bg4GVeZr|1x<5R`K}P32xSzwq)NmRx{hRI~9^&qb&N=lGzhgrBO;lWu)HB_}cmk*5(XzN>6<;lJGFco%K^Rvi^q9pHsd#9K"
    "3&`SZRNR2jx%gF_j?X+KOm$qd5kHK?d$tMVafDu-qFYJh-nsE2bSHEw^1&;<?-YI#)qkqqBlH(5IbieD_llnN`=s0`+$1Ws3MU#b"
    "TAn$DuM9sbO*D><4)Ky4nOkmnB^R>ioR8J>nDG+{UV4OhvE3D}x`%Q@=-+D`aT=R2UMP!aQ~1i_Z%i1ErsIfW6@RAV-?BI$6ULue"
    "6=x&urCjPs{7oWqqeCc%QXLQ7E{wy;;ye`}_j&lcQ*@fqcOS*4m40NEJTZN8<0Vma=M>$FUS}0ww@QyQq8}ZdTmQ=FwpU26(oV`V"
    "zAyb8t>gr!UeUEx^c^X=;CqC6P|_om6Iov9_t21@MXzy+j$<Vkgg&KQ__OFgR^0=T-7BD;r(JT(jVwN`bEp@^>OKpJ8xy+B;@CQb"
    "@<r%9QE|@to(rK{p?jfoT`uus5^tI0iLVs-k*fH9iNp8G-bfUGHcIcw_^+l%g(ty5oX4IpdD5ThHARYFEBWPm<+c}uuL~~3evtm-"
    "UdmojazN5p+ZRWvdI~3^FNuC7^1&#ah`k|rv79=^|0SO?J`za}4ebQcuTq5{u@{m&`cx|T=oaE9*7}aZO`_ud#eNX}EqKWEq)wrJ"
    "WK=wVR=;zI6OjW(-FG4Slcd?bNa`N3NX739KI9%UiMy0IWO*)el2-A3q3>AfGul3t=TUKPx{nktoT7J)8!aCaHLkc<ZuyeoK+6eb"
    "&r3a#D;hsW$(88n+~pZAjtu3AwlBo)PL$p4m41}j9X&&w2tGtFN)`Q!JP|wFDSJ7e-7TraQKt$wQf`&KEpf>v&;4VmzSl|FC5f`D"
    "GkvWflt0-$Npk;%+z%uBw{qWwx=&2-k;VTPDEZM*;jCSV2Yr8q+>4YH=eDm?rMEaGSG0Yu{ZzhpZu>vV<EJu8AG4~x#+BB;wA_e_"
    "{=1!?6y>%bY##X_sZ;uu=vC&(klsbVG9oW}hIY17c7xYAQ1U|bs^qZHUeJCd(XXt+L3VG<p~`+%@&2j0w<0U=mK!(X_c7`o7q9Nq"
    "6Py@zKY&y6L6-|IBpzS<N`jkM$q%FA^yNOWSlufo_P6L&a$X|4H==FGw@p!QxgmN|=08$&oXL|s_CaQE_fmS4dXL6olxIA`|6}dH"
    "Rqc$DL#g)P_R4LSIHgxb3Ri-USmDFv(X#{}wr7Y7(W??=cR1yjlILP~XZln35GR?PacCa=7{%YahI&_|?$J!uJvRFO%Z%?Uf0VXc"
    "L~g|D-pNGWYmwbQsr*sdJxR*mw(5Q)tLQ#b_l{+9L*Jt$_rj?A^u!KHigN2oN$))S2#T*~_{!&(DhmCwR@*DeF1JOw%k}uuqTK$h"
    "RQ03xHN+lImHw3|{Vh_sc1J4vUC9xn^enMIQo+mNp<bo+Ez=|SK1Q~GacHP_B}%`NRO}7Wm%Q4aNL2aH?zdESNOmunzL&CVXorj5"
    "l_)zU^EY);_Dtu{4iR~f-JhiVOIF{H)G@Tr6O|V`N8fjx-3KG%QvO}J&n2_pi*x(0^!}YM%DwN9*&#~4XniWmr)TN?MrDu0+P|vo"
    ">saYmvGQAG?<orRClb96qU1-U>=LW?^+bM%9pco!hEw}_u}ZZ+E9sHjE*JYF>Ym%q5P4zMzMjx`qVOVeBvt!(Uiq;^kILuQP1XH+"
    "R^6i%DSIXJt91<Rm+YQmwSSk{8685o<MjQmofMALeXiO4-=deP`zVdNmo<CdR?%lsZvJf4KETiHZ-pnT_US}FbIK2<_XACyeR#cZ"
    "pD1}?mEGYKzT`cwIJaFBtM9j}+^D>^uk`r=z3*c4_&sDiNpWs@sO>$yFYA;(YSj89ioay`S#fTEpZ34#eP5^cqarmAu{w8<sB;fa"
    "k5AeuKWCiBpOfU#8}#{zRDGw@`gu`q`c3nUPt!wYFXYj)e4g_qF3*01_7__<uQt#Aq_+FDoO5~n2T30JmTzB1pQF+GnJfP9a?`4Q"
    ")84uDN1G@AGr8D1_r8%U%DpZwpPVkrEth@o-1|eR!j1N8YChuCd}%*@(ku6RrbW4NrTsN|>0Zr8rmracI>p~y@7(y+{iS*Qtor>v"
    "&pt$?>SuaTuiSQurdMB_dmhp}`;sOPJ!m`TXXmLjef7@mFUThkY;kV?UR0cWT~kGuUZqa)b<JPm;yifnom;MGI@J5TM#)R7=vA+)"
    "K8N7*_%{;GuZnWhpVj+j#s3{AKRc(_JNLeW>64p2wLG$FJv2X!^geZQZolo%_Q|z=qvlidmux>-*_)}-i<3O|pjZ2+Uiq`V(hD>H"
    "8{ZR|O5Bp%&n@>#%6+SHU%lKj$#FFC#xO1^=D3}hpWD@8yimmPM3MY&gv3K}e3RjL%tYdJ<bLB>9Csrr$LUy(n~LRL%Ng8`>&4Qq"
    "#3ylFkNKQaxxbv>o37&VaeS1MI4ORQE9ZVWek=Wv-zWFeO1#x%P9<)M-<M7b=O>bQEAGef{3H_J#O=M@1F|)Y--<c@2yuq{Q;DbI"
    "xH8Y<wcL;4_sIH1oVW75K8#B<{63)%%kf@@*V(V<RP~c^JXiX5Nr0Q5?NjtodcNLwK1%uddxm~Vu^YX*AJ?jVn(Y2$xyM-UnHPE#"
    "y5=}9i4zn0@D*HdMYvvx#Qkx+Te2m*U)6jO#^<G+k2!Tr|M4`g$MI^33+YdAA?px-7~)9c(_}q8zt3(8zsK@=S&3ugehkxd#B>{T"
    "J;U_mxt^WMDfb^S?xKAmJ)|G2_iYI0ArU&A7}8s`K8#<??p<fP6n>F%++w8T8ijA{<@Bd8KGAcWU%W2FtzW|B+>fgc=TYJyv;VW<"
    "@y2_@`SQ$njKp=yJiZmqPbBLhX|LRRiPd&n@6cY){4_m6e~Qufn~UFo_a){3eH^c!*`wk$B`%)nIhFP1@Ab?0yP3W`&ySOSc82^{"
    "_*g1(MB-W{j#SoH`1-z3o+OIzSmBeKL%y1=8?R%uC*<Ek@0P#AF<oZ&^D|vUOqc0+9>;t9zO7*#YQ*uWZf+PqDtt5I@3Lz{zMb&="
    "TEPkT7q1NYyvW~(`MBJpp6&_xy~xX`O2zRm4EahrR?R!FqsXmD;+2^XI3B0l9L5(%9KRd$b5kA0>8AU_c;JZV-+mR+y~vH2`KlWi"
    "u2;rq^PHcG^?7l%pC)=$qRy3ib?(Qh{i0O;zl@wyll@KcgB$g~dGdb`a=*Bg%l`~XT)5y+?#GYRJ>b%>+<z}}M*inQ|6hdMGcNxN"
    "lZYNC@y}c@5kAQ4=0#3@r}(|-Jxq_O@HZI;uV1<|)Pt<VVGIABui~>ehw@JNv&c=ybm?X?-Fy(1oAqJbxLv@h(4pvQJf4X0oR)`t"
    "Je?5ggR#i%k3+d6a>gi}BtL}mSnw|65`9zjQ(qJ6mHy2zo?PTX^j;Wup79T!C)`7`|NBxE$FKIkBV}jla~!GKACrABr{fdUzO7g1"
    "J-yl&lJf~-cZl94_wuJqZ`uDSiaZy7DEikXb)Uh`a9%U{EBbq7xSo#rjMyzg=fbbjEg^psJ}K*$=?|i(GzjV5cy%wqlyIFy-xB<Z"
    "-Y4=(^Z+v>lxL!UnQue;LGb9dE4d)}na=G6=ORa$?%k)M-0?h5DX0GP@V<n&GPJ)^(TmoE_b*uC-z<O4_v-!!)~nKwINcQ9n~;b+"
    "c|V*7xqrfc8tR!w>=^#8gympVq4X-j)y7bMWc$W_a{Gmx!iU~3mi=KlHx#LJ5Bl7vSNl40PBZ(TR{0+Z`9Br;e<ArF)4GbU%ewIV"
    "$p1PBKazP9y(dxkcVv05_%g49S)%j-x%Wc!DxT+*=~3(_;lE;cia)?CR(^slAs^55lOIDpUhI6i|3v67`YP1B9rI-``p`U18CPy<"
    "sGo}6pYr#n(?h#A5<5cvXHDD7;%5;52;(6Z9Ig)a9kVXflQMmOLa2X=-H}u){aMy`TgYGhIyLUS;k*bAh0n>lh&?8HW~R62*>7+~"
    "x%ZbePV~N#oI^^~IqpQALy~iOa<0Ov^G0%hLZ3Sl`D8gS=TmK-bE<L<+^cgF**PYC&RcNWHuM|F|GNlZ%<k_IeM9}9kvtduC|(`P"
    "JHe&oV`Xn<>!bVt{)>=)Gk(4=lwW3AXtyVuLb;X6H|8^LU2Z!|_<{LB$t&e2i1&ncbF@m)o!qO#dXC=`-a{nwA@~2t_ei<elktX7"
    "E{gv_`gfuy%Ktm>3H8BvyYlmV7V5?6+R&e+{pL1LoQ7Ba?j+AS3!lf2mFRQN`v1vPT(DFB2PXd)m;FDZ_+^A&@;Zy(MC82P66(!j"
    "9}69bo-h0>+7!}r6ovjY;nzY3qK{;H^Y@{=7CkJp6L~y|x(`b9yol*v{y#689ny>FNtu6kZRlV1Uxxjs6`_AZ{5ayz5V{gQN9b0@"
    "lP(VRQ}F|2{)cZvJPVzfPs4dg#hwxU|088z{TTYwMSjbAir-ND05b1>bts>N?lb)Sq~80T@?(i#@xAc5_(k;j!hHL{kvhL;)p=I0"
    "&TnUNJSrY1JKrpJx{4DJJ6--C%qsm#_Bljf7I`oHOXxq-6Ly4lT;@*|xi5CR!j04R+n!M0@^wPHMEsPQ9?bM?#ox9(Tqn_UWxO)}"
    "%+6h+^yF>I4rlpcMPCyC7wb`(ykj|%)+qW|9gZ_%yqhhd9iMIu^>^_Ph@VRMh2i?*zwq0``_{zHNhd40xhK>!MK8|ica0kNHnm^1"
    "KIHde-+9K9{NJGX2gKeJzG&8ja#;AI6@H=RM!xg4>YjpF>rp*I`$O~{!9{kiOvek$xk-^9a$ZdS|5W~exK4-*kq1WX+N#k05IbJ@"
    "pzM3d|N07_lzS0nJ%!I_{zK8<S1Y}O^{G_k^uEv^E&jpmKENMByCmCB5uAS$`nkmqo+`aT=+k`{;vt<9+TYSX7C(&G-@+$E51Jk7"
    "`?3#e=Z5|aNu$-_{+?M7?jvS$S^N}}l)q`S^82!!vub}cDG&7kS-;5T_Ad(#MPJPJpB9AcBl@Znf7-hx!Mr3ZGkN@fH=aD#"
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

