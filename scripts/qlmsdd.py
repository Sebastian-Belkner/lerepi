import sys
import lerepi.params.90b91 as par

# sim_id = int(sys.argv[1])
for sim_id in range(200):
    par.qlms_dd.get_sim_qlm('p_p', sim_id)
    print('{} done.'.format(sim_id))