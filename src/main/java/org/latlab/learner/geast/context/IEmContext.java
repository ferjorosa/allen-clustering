package org.latlab.learner.geast.context;

import org.latlab.learner.geast.EmFramework;

import java.util.concurrent.Executor;

public interface IEmContext {

	public EmFramework screeningEm();

	public EmFramework selectionEm();

	public Executor searchExecutor();

}
